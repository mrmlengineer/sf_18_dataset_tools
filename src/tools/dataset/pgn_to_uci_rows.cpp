/*
  Convert PGN games to compact UCI mainline TSV rows.

  Output format:
    <min_elo_or_empty>\t<uci1,uci2,...>
*/

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if HAVE_ZLIB
#include <zlib.h>
#endif

#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include "types.h"

namespace {

constexpr const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Options {
    std::string inPath;
    std::string outPath = "-";
    std::size_t maxRows = 0;
    double progressEverySeconds = 30.0;
    std::size_t threads = 1;
    std::size_t batchGames = 256;
};

struct Stats {
    std::size_t gamesParsed = 0;
    std::size_t rowsWritten = 0;
    std::size_t skippedSetupOrFen = 0;
    std::size_t skippedEmptyMainline = 0;
    std::size_t parseErrors = 0;
    std::size_t nullMinEloRows = 0;
    std::size_t movesWritten = 0;
};

struct Game {
    std::map<std::string, std::string> headers;
    std::string movetext;
};

struct GameResult {
    std::optional<int> minElo;
    std::vector<std::string> moves;
    std::string error;
    bool rowWritten = false;
    bool skippedSetupOrFen = false;
    bool skippedEmptyMainline = false;
    bool parseError = false;
    bool nullMinElo = false;
};

struct BatchWork {
    std::size_t id = 0;
    std::size_t firstGameNumber = 0;
    std::vector<Game> games;
};

struct BatchResult {
    std::size_t id = 0;
    std::size_t firstGameNumber = 0;
    std::vector<GameResult> games;
};

class LineReader {
   public:
    explicit LineReader(const std::string& path) {
        if (ends_with(path, ".gz"))
        {
#if HAVE_ZLIB
            gz_ = gzopen(path.c_str(), "rb");
            if (!gz_)
                throw std::runtime_error("Could not open gzip input: " + path);
#else
            const std::string command = "gzip -cd -- " + shell_quote(path);
            pipe_ = popen(command.c_str(), "r");
            if (!pipe_)
                throw std::runtime_error("Could not open gzip input via gzip command: " + path);
#endif
        }
        else
        {
            file_.open(path);
            if (!file_)
                throw std::runtime_error("Could not open input: " + path);
        }
    }

    bool getline(std::string& line) {
        line.clear();
#if HAVE_ZLIB
        if (gz_)
            return gz_getline(line);
#else
        if (pipe_)
            return pipe_getline(line);
#endif
        return static_cast<bool>(std::getline(file_, line));
    }

    ~LineReader() {
#if HAVE_ZLIB
        if (gz_)
            gzclose(gz_);
#else
        if (pipe_)
            pclose(pipe_);
#endif
    }

   private:
    static std::string shell_quote(std::string_view text) {
        std::string quoted = "'";
        for (const char ch : text)
        {
            if (ch == '\'')
                quoted += "'\\''";
            else
                quoted.push_back(ch);
        }
        quoted.push_back('\'');
        return quoted;
    }

    static bool ends_with(std::string_view value, std::string_view suffix) {
        return value.size() >= suffix.size()
            && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

#if HAVE_ZLIB
    bool gz_getline(std::string& line) {
        char buffer[1 << 15];
        while (true)
        {
            char* result = gzgets(gz_, buffer, static_cast<int>(sizeof(buffer)));
            if (!result)
            {
                if (gzerror(gz_, nullptr) != Z_OK && !gzeof(gz_))
                    throw std::runtime_error("Error while reading gzip input");
                return !line.empty();
            }

            line += buffer;
            if (!line.empty() && line.back() == '\n')
            {
                line.pop_back();
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                return true;
            }
        }
    }

    gzFile gz_ = nullptr;
#else
    bool pipe_getline(std::string& line) {
        char buffer[1 << 15];
        while (true)
        {
            char* result = std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe_);
            if (!result)
            {
                if (std::ferror(pipe_))
                    throw std::runtime_error("Error while reading gzip input");
                return !line.empty();
            }

            line += buffer;
            if (!line.empty() && line.back() == '\n')
            {
                line.pop_back();
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                return true;
            }
        }
    }

    FILE* pipe_ = nullptr;
#endif
    std::ifstream file_;
};

std::string trim_copy(std::string_view text) {
    std::size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first])))
        ++first;
    std::size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1])))
        --last;
    return std::string(text.substr(first, last - first));
}

bool starts_with_bom(std::string_view line) {
    return line.size() >= 3
        && static_cast<unsigned char>(line[0]) == 0xEF
        && static_cast<unsigned char>(line[1]) == 0xBB
        && static_cast<unsigned char>(line[2]) == 0xBF;
}

bool parse_tag_pair(const std::string& line, std::string& tag, std::string& value) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.size() < 4 || trimmed.front() != '[' || trimmed.back() != ']')
        return false;

    std::size_t pos = 1;
    while (pos < trimmed.size() - 1
           && (std::isalnum(static_cast<unsigned char>(trimmed[pos])) || trimmed[pos] == '_'))
        ++pos;
    if (pos == 1 || pos >= trimmed.size() - 1 || !std::isspace(static_cast<unsigned char>(trimmed[pos])))
        return false;

    tag = trimmed.substr(1, pos - 1);
    while (pos < trimmed.size() - 1 && std::isspace(static_cast<unsigned char>(trimmed[pos])))
        ++pos;
    if (pos >= trimmed.size() - 1 || trimmed[pos] != '"')
        return false;
    ++pos;

    std::string parsed;
    bool escaped = false;
    for (; pos < trimmed.size() - 1; ++pos)
    {
        const char ch = trimmed[pos];
        if (escaped)
        {
            parsed.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\')
        {
            escaped = true;
            continue;
        }
        if (ch == '"')
        {
            ++pos;
            while (pos < trimmed.size() - 1 && std::isspace(static_cast<unsigned char>(trimmed[pos])))
                ++pos;
            if (pos != trimmed.size() - 1)
                return false;
            value = std::move(parsed);
            return true;
        }
        parsed.push_back(ch);
    }

    return false;
}

void absorb_line(Game& game, const std::string& line) {
    std::string tag;
    std::string value;
    if (parse_tag_pair(line, tag, value))
        game.headers[tag] = value;
    else
    {
        game.movetext += line;
        game.movetext.push_back('\n');
    }
}

class PgnGameReader {
   public:
    explicit PgnGameReader(const std::string& path) :
        reader_(path) {}

    bool next(Game& out) {
        out = Game();
        std::string line;

        if (hasPending_)
        {
            absorb_line(out, pendingLine_);
            hasPending_ = false;
        }

        bool haveAny = !out.headers.empty() || !out.movetext.empty();
        bool seenMovetext = !out.movetext.empty();
        bool firstLine = !started_;

        while (reader_.getline(line))
        {
            started_ = true;
            if (firstLine)
            {
                firstLine = false;
                if (starts_with_bom(line))
                    line.erase(0, 3);
            }

            std::string tag;
            std::string value;
            const bool isHeader = parse_tag_pair(line, tag, value);
            if (isHeader && haveAny && seenMovetext)
            {
                pendingLine_ = line;
                hasPending_ = true;
                return true;
            }

            if (isHeader)
                out.headers[tag] = value;
            else
            {
                out.movetext += line;
                out.movetext.push_back('\n');
                if (!trim_copy(line).empty())
                    seenMovetext = true;
            }
            haveAny = true;
        }

        return haveAny;
    }

   private:
    LineReader reader_;
    std::string pendingLine_;
    bool hasPending_ = false;
    bool started_ = false;
};

std::optional<int> parse_elo(const std::string* value) {
    if (!value)
        return std::nullopt;
    const std::string trimmed = trim_copy(*value);
    if (trimmed.empty())
        return std::nullopt;
    std::size_t pos = 0;
    try
    {
        int parsed = std::stoi(trimmed, &pos);
        if (pos != trimmed.size())
            return std::nullopt;
        return parsed;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

const std::string* header_value(const Game& game, const std::string& key) {
    const auto it = game.headers.find(key);
    return it == game.headers.end() ? nullptr : &it->second;
}

std::optional<int> min_elo_or_none(const Game& game) {
    const std::optional<int> white = parse_elo(header_value(game, "WhiteElo"));
    const std::optional<int> black = parse_elo(header_value(game, "BlackElo"));
    if (!white || !black)
        return std::nullopt;
    return std::min(*white, *black);
}

bool is_result_token(const std::string& token) {
    return token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*";
}

std::string strip_move_number_prefix(std::string token) {
    while (!token.empty())
    {
        std::size_t pos = 0;
        while (pos < token.size() && std::isdigit(static_cast<unsigned char>(token[pos])))
            ++pos;
        if (pos == 0 || pos >= token.size() || token[pos] != '.')
            break;
        while (pos < token.size() && token[pos] == '.')
            ++pos;
        token.erase(0, pos);
    }
    return token;
}

bool is_move_number_marker(const std::string& token) {
    if (token.empty())
        return false;
    bool hasDot = false;
    for (const char ch : token)
    {
        if (ch == '.')
        {
            hasDot = true;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(ch)))
            return false;
    }
    return hasDot;
}

bool is_annotation_marker(const std::string& token) {
    if (token.empty())
        return false;
    for (const char ch : token)
        if (ch != '!' && ch != '?')
            return false;
    return true;
}

std::vector<std::string> tokenize_movetext(const std::string& movetext) {
    std::vector<std::string> tokens;
    std::string current;
    int variationDepth = 0;
    bool inComment = false;
    bool inLineComment = false;

    auto flush = [&]() {
        if (!current.empty())
        {
            std::string token = strip_move_number_prefix(current);
            if (!token.empty() && token.front() == '$')
                token.clear();
            if (!token.empty() && !is_move_number_marker(token) && !is_annotation_marker(token)
                && !is_result_token(token))
                tokens.push_back(token);
            current.clear();
        }
    };

    for (char ch : movetext)
    {
        if (inLineComment)
        {
            if (ch == '\n' || ch == '\r')
                inLineComment = false;
            continue;
        }
        if (inComment)
        {
            if (ch == '}')
                inComment = false;
            continue;
        }
        if (variationDepth > 0)
        {
            if (ch == '{')
                inComment = true;
            else if (ch == ';')
                inLineComment = true;
            else if (ch == '(')
                ++variationDepth;
            else if (ch == ')')
                --variationDepth;
            continue;
        }

        if (ch == '{')
        {
            flush();
            inComment = true;
            continue;
        }
        if (ch == ';')
        {
            flush();
            inLineComment = true;
            continue;
        }
        if (ch == '(')
        {
            flush();
            variationDepth = 1;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            flush();
            continue;
        }
        current.push_back(ch);
    }
    flush();
    return tokens;
}

std::string normalize_san(std::string token) {
    while (!token.empty())
    {
        const char ch = token.back();
        if (ch == '+' || ch == '#' || ch == '!' || ch == '?')
            token.pop_back();
        else
            break;
    }
    while (!token.empty() && (token.back() == '\r' || token.back() == '\n'))
        token.pop_back();
    return token;
}

Stockfish::PieceType piece_type_from_char(char ch) {
    using namespace Stockfish;
    switch (ch)
    {
    case 'N' :
        return KNIGHT;
    case 'B' :
        return BISHOP;
    case 'R' :
        return ROOK;
    case 'Q' :
        return QUEEN;
    case 'K' :
        return KING;
    default :
        return PAWN;
    }
}

Stockfish::PieceType promotion_from_char(char ch) {
    return piece_type_from_char(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
}

bool is_file_char(char ch) {
    return ch >= 'a' && ch <= 'h';
}

bool is_rank_char(char ch) {
    return ch >= '1' && ch <= '8';
}

Stockfish::Square square_from_chars(char file, char rank) {
    return Stockfish::make_square(Stockfish::File(file - 'a'), Stockfish::Rank(rank - '1'));
}

std::string square_to_string(Stockfish::Square sq) {
    return std::string{char('a' + Stockfish::file_of(sq)), char('1' + Stockfish::rank_of(sq))};
}

std::string move_to_uci(Stockfish::Move move, bool chess960) {
    using namespace Stockfish;
    Square from = move.from_sq();
    Square to = move.to_sq();
    if (move.type_of() == CASTLING && !chess960)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    std::string result = square_to_string(from) + square_to_string(to);
    if (move.type_of() == PROMOTION)
        result.push_back(" pnbrqk"[move.promotion_type()]);
    return result;
}

struct ParsedSan {
    Stockfish::PieceType piece = Stockfish::NO_PIECE_TYPE;
    Stockfish::Square to = Stockfish::SQ_NONE;
    Stockfish::PieceType promotion = Stockfish::NO_PIECE_TYPE;
    std::optional<Stockfish::File> fromFile;
    std::optional<Stockfish::Rank> fromRank;
    bool capture = false;
    bool castling = false;
    bool kingSideCastle = false;
};

std::optional<ParsedSan> parse_san_token(std::string token) {
    using namespace Stockfish;

    token = normalize_san(strip_move_number_prefix(std::move(token)));
    if (token.empty() || is_result_token(token))
        return std::nullopt;
    if (token == "--" || token == "Z0")
        return std::nullopt;

    if (token == "O-O" || token == "0-0")
        return ParsedSan{KING, SQ_NONE, NO_PIECE_TYPE, std::nullopt, std::nullopt, false, true, true};
    if (token == "O-O-O" || token == "0-0-0")
        return ParsedSan{KING, SQ_NONE, NO_PIECE_TYPE, std::nullopt, std::nullopt, false, true, false};

    ParsedSan parsed;
    parsed.piece = piece_type_from_char(token.front());
    std::size_t start = parsed.piece == PAWN ? 0 : 1;
    parsed.capture = token.find('x') != std::string::npos;

    std::size_t promotionPos = token.find('=');
    if (promotionPos != std::string::npos && promotionPos + 1 < token.size())
    {
        parsed.promotion = promotion_from_char(token[promotionPos + 1]);
        token.erase(promotionPos);
    }
    else if (parsed.piece == PAWN && token.size() >= 3)
    {
        const char maybePromotion = token.back();
        if ((maybePromotion == 'N' || maybePromotion == 'B' || maybePromotion == 'R' || maybePromotion == 'Q')
            && is_file_char(token[token.size() - 3]) && is_rank_char(token[token.size() - 2]))
        {
            parsed.promotion = promotion_from_char(maybePromotion);
            token.pop_back();
        }
    }

    std::string core;
    for (std::size_t i = start; i < token.size(); ++i)
        if (token[i] != 'x')
            core.push_back(token[i]);

    if (core.size() < 2 || !is_file_char(core[core.size() - 2]) || !is_rank_char(core[core.size() - 1]))
        return std::nullopt;

    parsed.to = square_from_chars(core[core.size() - 2], core[core.size() - 1]);
    const std::string disambiguation = core.substr(0, core.size() - 2);
    for (char ch : disambiguation)
    {
        if (is_file_char(ch))
            parsed.fromFile = Stockfish::File(ch - 'a');
        else if (is_rank_char(ch))
            parsed.fromRank = Stockfish::Rank(ch - '1');
        else
            return std::nullopt;
    }

    return parsed;
}

bool from_disambiguation_matches(Stockfish::Move move, const ParsedSan& san) {
    if (san.fromFile && Stockfish::file_of(move.from_sq()) != *san.fromFile)
        return false;
    if (san.fromRank && Stockfish::rank_of(move.from_sq()) != *san.fromRank)
        return false;
    return true;
}

Stockfish::Move resolve_san(const Stockfish::Position& pos, const std::string& token) {
    using namespace Stockfish;
    const std::optional<ParsedSan> san = parse_san_token(token);
    if (!san)
        return Move::none();

    Move matched = Move::none();
    int matches = 0;

    for (const Move move : MoveList<LEGAL>(pos))
    {
        const Piece movedPiece = pos.piece_on(move.from_sq());
        if (type_of(movedPiece) != san->piece)
            continue;

        if (san->castling)
        {
            if (move.type_of() != CASTLING)
                continue;
            const std::string uci = move_to_uci(move, pos.is_chess960());
            const bool kingSide = uci.size() >= 4 && (uci[2] == 'g');
            if (kingSide != san->kingSideCastle)
                continue;
        }
        else
        {
            if (move.type_of() == CASTLING)
                continue;
            if (move.to_sq() != san->to)
                continue;
            if (pos.capture(move) != san->capture)
                continue;
            if (san->promotion != NO_PIECE_TYPE)
            {
                if (move.type_of() != PROMOTION || move.promotion_type() != san->promotion)
                    continue;
            }
            else if (move.type_of() == PROMOTION)
                continue;
            if (!from_disambiguation_matches(move, *san))
                continue;
        }

        matched = move;
        ++matches;
    }

    return matches == 1 ? matched : Move::none();
}

bool convert_game_moves(const Game& game, std::vector<std::string>& uciMoves, std::string& error) {
    using namespace Stockfish;
    uciMoves.clear();
    const std::vector<std::string> sanTokens = tokenize_movetext(game.movetext);
    if (sanTokens.empty())
        return true;

    Position pos;
    std::deque<StateInfo> states;
    states.emplace_back();
    pos.set(StartFEN, false, &states.back());

    for (const std::string& token : sanTokens)
    {
        Move move = resolve_san(pos, token);
        if (move == Move::none())
        {
            error = "Could not resolve SAN token '" + token + "'";
            return false;
        }
        uciMoves.push_back(move_to_uci(move, pos.is_chess960()));
        states.emplace_back();
        pos.do_move(move, states.back());
    }

    return true;
}

void print_usage() {
    std::cerr
      << "Usage:\n"
      << "  pgn_to_uci_rows --in <input.pgn[.gz]> [--out <path|-] [--max-rows <N>]\n"
      << "                     [--progress-every-seconds <seconds>] [--threads <N>]\n"
      << "                     [--batch-games <N>]\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--in" && i + 1 < argc)
            opt.inPath = argv[++i];
        else if (arg == "--out" && i + 1 < argc)
            opt.outPath = argv[++i];
        else if (arg == "--max-rows" && i + 1 < argc)
            opt.maxRows = std::stoull(argv[++i]);
        else if (arg == "--progress-every-seconds" && i + 1 < argc)
            opt.progressEverySeconds = std::stod(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)
            opt.threads = std::stoull(argv[++i]);
        else if (arg == "--batch-games" && i + 1 < argc)
            opt.batchGames = std::stoull(argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            print_usage();
            std::exit(0);
        }
        else
            return false;
    }
    opt.threads = std::max<std::size_t>(1, opt.threads);
    opt.batchGames = std::max<std::size_t>(1, opt.batchGames);
    opt.progressEverySeconds = std::max(0.0, opt.progressEverySeconds);
    return !opt.inPath.empty();
}

void emit_row(std::ostream& out, const std::optional<int>& minElo, const std::vector<std::string>& moves) {
    if (minElo)
        out << *minElo;
    out << '\t';
    for (std::size_t i = 0; i < moves.size(); ++i)
    {
        if (i)
            out << ',';
        out << moves[i];
    }
    out << '\n';
}

void print_progress(const Stats& stats, const std::chrono::steady_clock::time_point& startedAt) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    const double rate = elapsed > 0.0 ? static_cast<double>(stats.gamesParsed) / elapsed : 0.0;
    std::cerr << "[pgn_to_uci_rows] parsed " << stats.gamesParsed
              << " games | kept " << stats.rowsWritten
              << " rows | rate " << rate << " games/s\n";
}

void maybe_print_progress(const Options& opt,
                          const Stats& stats,
                          const std::chrono::steady_clock::time_point& startedAt,
                          std::chrono::steady_clock::time_point& lastProgressAt) {
    if (stats.gamesParsed == 0)
        return;

    const auto now = std::chrono::steady_clock::now();
    if (opt.progressEverySeconds <= 0.0
        || std::chrono::duration<double>(now - lastProgressAt).count() < opt.progressEverySeconds)
        return;

    print_progress(stats, startedAt);
    lastProgressAt = now;
}

void print_summary(const Stats& stats, const std::chrono::steady_clock::time_point& startedAt) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    std::cerr << "[pgn_to_uci_rows] summary"
              << " games_parsed=" << stats.gamesParsed
              << " rows_written=" << stats.rowsWritten
              << " skipped_setup_or_fen=" << stats.skippedSetupOrFen
              << " skipped_empty_mainline=" << stats.skippedEmptyMainline
              << " parse_errors=" << stats.parseErrors
              << " null_min_elo_rows=" << stats.nullMinEloRows
              << " moves_written=" << stats.movesWritten
              << " elapsed_seconds=" << elapsed
              << "\n";
}

GameResult process_game(const Game& game) {
    GameResult result;
    const std::string* setup = header_value(game, "SetUp");
    const std::string* fen = header_value(game, "FEN");
    if ((setup && trim_copy(*setup) == "1") || (fen && !trim_copy(*fen).empty()))
    {
        result.skippedSetupOrFen = true;
        return result;
    }

    if (!convert_game_moves(game, result.moves, result.error))
    {
        result.parseError = true;
        return result;
    }
    if (result.moves.empty())
    {
        result.skippedEmptyMainline = true;
        return result;
    }

    result.minElo = min_elo_or_none(game);
    result.nullMinElo = !result.minElo;
    result.rowWritten = true;
    return result;
}

BatchResult process_batch(BatchWork work) {
    BatchResult result;
    result.id = work.id;
    result.firstGameNumber = work.firstGameNumber;
    result.games.reserve(work.games.size());
    for (const Game& game : work.games)
        result.games.push_back(process_game(game));
    return result;
}

void accumulate_non_row_stats(Stats& stats, const GameResult& result) {
    if (result.skippedSetupOrFen)
        ++stats.skippedSetupOrFen;
    if (result.skippedEmptyMainline)
        ++stats.skippedEmptyMainline;
    if (result.parseError)
        ++stats.parseErrors;
}

void write_game_result(std::ostream& out,
                       Stats& stats,
                       const GameResult& result,
                       std::size_t gameNumber,
                       bool canWriteRows) {
    ++stats.gamesParsed;
    accumulate_non_row_stats(stats, result);

    if (result.parseError)
        std::cerr << "[pgn_to_uci_rows] parse error in game " << gameNumber
                  << ": " << result.error << "\n";

    if (result.rowWritten && canWriteRows)
    {
        if (result.nullMinElo)
            ++stats.nullMinEloRows;
        emit_row(out, result.minElo, result.moves);
        ++stats.rowsWritten;
        stats.movesWritten += result.moves.size();
    }
}

void write_batch_result(std::ostream& out,
                        Stats& stats,
                        const BatchResult& batch,
                        const Options& opt,
                        const std::chrono::steady_clock::time_point& startedAt,
                        std::chrono::steady_clock::time_point& lastProgressAt,
                        bool& rowLimitReached) {
    for (std::size_t i = 0; i < batch.games.size(); ++i)
    {
        if (rowLimitReached)
            break;
        const bool canWriteRows = !rowLimitReached;
        write_game_result(out, stats, batch.games[i], batch.firstGameNumber + i, canWriteRows);
        if (opt.maxRows > 0 && stats.rowsWritten >= opt.maxRows)
            rowLimitReached = true;

        maybe_print_progress(opt, stats, startedAt, lastProgressAt);
    }
}

int run(const Options& opt) {
    using namespace Stockfish;
    Bitboards::init();
    Position::init();

    std::ofstream fileOut;
    std::ostream* out = &std::cout;
    if (opt.outPath != "-")
    {
        fileOut.open(opt.outPath);
        if (!fileOut)
            throw std::runtime_error("Could not open output: " + opt.outPath);
        out = &fileOut;
    }

    PgnGameReader reader(opt.inPath);
    Stats stats;
    const auto startedAt = std::chrono::steady_clock::now();
    auto lastProgressAt = startedAt;

    if (opt.threads <= 1)
    {
        Game game;
        bool rowLimitReached = false;
        while (!rowLimitReached && reader.next(game))
        {
            const GameResult result = process_game(game);
            write_game_result(*out, stats, result, stats.gamesParsed + 1, true);

            maybe_print_progress(opt, stats, startedAt, lastProgressAt);
            if (opt.maxRows > 0 && stats.rowsWritten >= opt.maxRows)
                rowLimitReached = true;
        }
    }
    else
    {
        std::mutex mutex;
        std::condition_variable workCv;
        std::condition_variable resultCv;
        std::condition_variable spaceCv;
        std::deque<BatchWork> workQueue;
        std::map<std::size_t, BatchResult> results;
        std::exception_ptr workerException;
        bool readingDone = false;

        const std::size_t maxQueuedBatches = std::max<std::size_t>(2, opt.threads * 2);

        auto worker = [&]() {
            while (true)
            {
                BatchWork work;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    workCv.wait(lock, [&]() { return readingDone || !workQueue.empty(); });
                    if (workQueue.empty())
                        return;
                    work = std::move(workQueue.front());
                    workQueue.pop_front();
                    spaceCv.notify_one();
                }

                try
                {
                    BatchResult result = process_batch(std::move(work));
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        results.emplace(result.id, std::move(result));
                    }
                    resultCv.notify_one();
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!workerException)
                        workerException = std::current_exception();
                    readingDone = true;
                    workCv.notify_all();
                    resultCv.notify_all();
                    return;
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(opt.threads);
        for (std::size_t i = 0; i < opt.threads; ++i)
            workers.emplace_back(worker);

        auto take_next_result = [&](std::size_t nextBatchToWrite, bool wait) -> std::optional<BatchResult> {
            std::unique_lock<std::mutex> lock(mutex);
            if (wait)
                resultCv.wait(lock, [&]() {
                    return workerException || results.find(nextBatchToWrite) != results.end();
                });
            auto it = results.find(nextBatchToWrite);
            if (it == results.end())
                return std::nullopt;
            BatchResult result = std::move(it->second);
            results.erase(it);
            spaceCv.notify_one();
            return result;
        };

        std::size_t nextBatchId = 0;
        std::size_t nextBatchToWrite = 0;
        std::size_t batchesSubmitted = 0;
        std::size_t nextGameNumber = 1;
        bool rowLimitReached = false;
        Game game;
        BatchWork current;
        current.id = nextBatchId++;
        current.firstGameNumber = nextGameNumber;

        auto submit_current_batch = [&]() {
            if (current.games.empty())
                return;
            {
                std::unique_lock<std::mutex> lock(mutex);
                spaceCv.wait(lock, [&]() {
                    return workerException || workQueue.size() < maxQueuedBatches;
                });
                if (workerException)
                    std::rethrow_exception(workerException);
                workQueue.push_back(std::move(current));
                ++batchesSubmitted;
            }
            workCv.notify_one();
            current = BatchWork{};
            current.id = nextBatchId++;
            current.firstGameNumber = nextGameNumber;
        };

        while (!rowLimitReached && reader.next(game))
        {
            current.games.push_back(std::move(game));
            ++nextGameNumber;
            if (current.games.size() >= opt.batchGames)
                submit_current_batch();

            while (true)
            {
                auto ready = take_next_result(nextBatchToWrite, false);
                if (!ready)
                    break;
                write_batch_result(*out, stats, *ready, opt, startedAt, lastProgressAt, rowLimitReached);
                ++nextBatchToWrite;
            }
        }

        if (!rowLimitReached)
            submit_current_batch();
        {
            std::lock_guard<std::mutex> lock(mutex);
            readingDone = true;
        }
        workCv.notify_all();

        while (nextBatchToWrite < batchesSubmitted)
        {
            auto ready = take_next_result(nextBatchToWrite, true);
            if (workerException)
                std::rethrow_exception(workerException);
            if (!ready)
                break;
            write_batch_result(*out, stats, *ready, opt, startedAt, lastProgressAt, rowLimitReached);
            ++nextBatchToWrite;
        }

        for (std::thread& thread : workers)
            thread.join();
        if (workerException)
            std::rethrow_exception(workerException);
    }

    print_summary(stats, startedAt);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        print_usage();
        return 2;
    }

    try
    {
        return run(opt);
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Error: " << exc.what() << "\n";
        return 1;
    }
}

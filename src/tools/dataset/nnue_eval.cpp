/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "bitboard.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/network.h"
#include "position.h"
#include "tools/dataset/nnue_eval_shared.h"
#include "types.h"
#include "uci.h"

namespace {

constexpr const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct Options {
    std::vector<std::string> movesPaths;
    std::string fensPath;
    std::string outPath;
    bool        appendOut       = false;
    std::string dedupCachePath;
    std::string bigEvalFile;
    std::string smallEvalFile;
    enum class OutputNet {
        Big,
        Small
    };
    OutputNet   outputNet       = OutputNet::Big;
    int         fromPly         = 17;
    int         toPly           = 0;
    std::size_t maxPositions    = 0;
    std::size_t threads         = 1;
    bool        dedup           = true;
    std::size_t hashBits        = 64;
    bool        dedupReserve    = false;
    std::size_t dedupReserveCount = 0;
    bool        unscaledOutput  = false;
    nnue_eval_shared::NnueLabelMode nnueLabelMode = nnue_eval_shared::NnueLabelMode::Adjusted;
    std::size_t progressEvery   = 0;
    std::size_t flushEvery      = 1'000'000;
    std::size_t gamesTotalHint  = 0;
};

struct GameInput {
    std::string              startFen;
    std::vector<std::string> moves;
};

bool check_eval_file(const std::string& path, const char* label) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cerr << label << " file not found: " << path << "\n";
        return false;
    }

    const auto size = file.tellg();
    if (size >= 0 && size < 1024 * 1024)
        std::cerr << label << " file looks too small (" << size
                  << " bytes). It may be a git-lfs pointer.\n";

    return true;
}

std::string resolve_eval_file_path(const std::string& requestedPath,
                                   const std::string& binaryDir,
                                   const char*        defaultName) {
    namespace fs = std::filesystem;

    if (!requestedPath.empty())
        return requestedPath;

    const fs::path bin =
      binaryDir.empty() ? fs::current_path() : fs::path(binaryDir).lexically_normal();

    const std::array<fs::path, 4> candidates = {bin / defaultName,
                                                bin / ".." / "networks" / defaultName,
                                                fs::path("networks") / defaultName,
                                                fs::path("..") / "networks" / defaultName};

    for (const auto& p : candidates)
    {
        std::error_code ec;
        if (fs::exists(p, ec) && !ec)
            return p.lexically_normal().string();
    }

    return (bin / defaultName).lexically_normal().string();
}

void print_usage() {
    std::cerr
      << "Usage:\n"
      << "  nnue_eval (--moves <moves.txt> [--moves <moves2.txt> ...] | --fens <fens.txt>) [--out <out.csv>]\n"
      << "           [--append-out] [--dedup-cache <cache.bin>]\n"
      << "           [--big <evalfile>] [--small <evalfile>]\n"
      << "           [--net <big|small>]\n"
      << "           [--from-ply <N>] [--to-ply <N>] [--max-positions <N>]\n"
      << "           [--threads <N>] [--dedup|--no-dedup]\n"
      << "           [--hash-bits <64|128>]\n"
      << "           [--dedup-reserve <N>] [--no-dedup-reserve]\n"
      << "           [--output-scale <scaled|unscaled>] [--unscaled-output]\n"
      << "           [--nnue-label <adjusted|raw>]\n"
      << "           [--progress-every <rows>]\n"
      << "           [--flush-every <rows>]\n"
      << "           [--games-total-hint <N>]\n\n"
      << "Moves file format: one game per line, comma-separated UCI moves.\n"
      << "  Optional per-line start position: <FEN>|uci1,uci2,...\n"
      << "FEN file format: one FEN/EPD position per line (first 4 fields required).\n"
      << "--from-ply default: 17 (absolute ply from initial position)\n"
      << "--to-ply default: 0 (no upper ply limit)\n"
      << "--dedup default: on\n"
      << "--net default: big\n"
      << "--hash-bits default: 64\n"
      << "--output-scale default: scaled\n"
      << "--nnue-label default: adjusted\n"
      << "--progress-every default: 0 (disabled)\n"
      << "--flush-every default: 1000000 (0 disables periodic flush)\n"
      << "--games-total-hint default: 0 (unknown)\n"
      << "--dedup-reserve default: disabled (0 also disables)\n"
      << "--append-out: append to existing CSV (--out required)\n"
      << "--dedup-cache: persist/load dedup hashes for incremental runs\n"
      << "--dedup key: FEN board + side-to-move (ignores castling/ep/halfmove/fullmove)\n"
      << "Example:\n"
      << "  g1f3,c7c5\n"
      << "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|g1f3,c7c5\n"
      << "  nnue_eval --moves part1.csv --moves part2.csv --from-ply 17 --to-ply 100 --out out.csv\n"
      << "  nnue_eval --fens data/UHO_Lichess_4852_v1.epd --from-ply 0 --out out.csv\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--moves" && i + 1 < argc)
            opt.movesPaths.push_back(argv[++i]);
        else if ((arg == "--fens" || arg == "--epd") && i + 1 < argc)
            opt.fensPath = argv[++i];
        else if (arg == "--out" && i + 1 < argc)
            opt.outPath = argv[++i];
        else if (arg == "--append-out")
            opt.appendOut = true;
        else if (arg == "--dedup-cache" && i + 1 < argc)
            opt.dedupCachePath = argv[++i];
        else if (arg == "--big" && i + 1 < argc)
            opt.bigEvalFile = argv[++i];
        else if (arg == "--small" && i + 1 < argc)
            opt.smallEvalFile = argv[++i];
        else if (arg == "--net" && i + 1 < argc)
        {
            const std::string net = argv[++i];
            if (net == "big")
                opt.outputNet = Options::OutputNet::Big;
            else if (net == "small")
                opt.outputNet = Options::OutputNet::Small;
            else
                return false;
        }
        else if (arg == "--from-ply" && i + 1 < argc)
        {
            opt.fromPly = std::stoi(argv[++i]);
            if (opt.fromPly < 0)
                return false;
        }
        else if (arg == "--to-ply" && i + 1 < argc)
        {
            opt.toPly = std::stoi(argv[++i]);
            if (opt.toPly < 0)
                return false;
        }
        else if (arg == "--max-positions" && i + 1 < argc)
            opt.maxPositions = std::stoull(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)
            opt.threads = std::stoull(argv[++i]);
        else if (arg == "--dedup")
            opt.dedup = true;
        else if (arg == "--no-dedup")
            opt.dedup = false;
        else if (arg == "--hash-bits" && i + 1 < argc)
            opt.hashBits = std::stoull(argv[++i]);
        else if (arg == "--output-scale" && i + 1 < argc)
        {
            const std::string scale = argv[++i];
            if (scale == "scaled")
                opt.unscaledOutput = false;
            else if (scale == "unscaled")
                opt.unscaledOutput = true;
            else
                return false;
        }
        else if (arg == "--unscaled-output")
            opt.unscaledOutput = true;
        else if (arg == "--scaled-output")
            opt.unscaledOutput = false;
        else if (arg == "--nnue-label" && i + 1 < argc)
        {
            const std::string mode = argv[++i];
            if (mode == "adjusted")
                opt.nnueLabelMode = nnue_eval_shared::NnueLabelMode::Adjusted;
            else if (mode == "raw")
                opt.nnueLabelMode = nnue_eval_shared::NnueLabelMode::Raw;
            else
                return false;
        }
        else if (arg == "--progress-every" && i + 1 < argc)
            opt.progressEvery = std::stoull(argv[++i]);
        else if (arg == "--flush-every" && i + 1 < argc)
            opt.flushEvery = std::stoull(argv[++i]);
        else if (arg == "--games-total-hint" && i + 1 < argc)
            opt.gamesTotalHint = std::stoull(argv[++i]);
        else if (arg == "--dedup-reserve" && i + 1 < argc)
        {
            opt.dedupReserveCount = std::stoull(argv[++i]);
            opt.dedupReserve = opt.dedupReserveCount > 0;
        }
        else if (arg == "--no-dedup-reserve")
            opt.dedupReserve = false;
        else
            return false;
    }

    const bool hasMoves = !opt.movesPaths.empty();
    const bool hasFens  = !opt.fensPath.empty();
    if (hasMoves == hasFens || (opt.hashBits != 64 && opt.hashBits != 128))
        return false;

    if (opt.appendOut && opt.outPath.empty())
        return false;

    if (opt.toPly > 0 && opt.toPly < opt.fromPly)
        return false;

    return true;
}

std::string trim_ascii(const std::string& s) {
    auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
    auto begin = std::find_if_not(s.begin(), s.end(), is_space);
    auto end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
    if (begin >= end)
        return {};
    return std::string(begin, end);
}

std::string normalize_token(std::string token) {
    token = trim_ascii(token);
    if (token.empty())
        return token;

    if (token.front() == '[')
        token.erase(token.begin());
    if (!token.empty() && token.back() == ']')
        token.pop_back();

    token = trim_ascii(token);
    if (token.size() >= 2 && token.front() == '"' && token.back() == '"')
        return token.substr(1, token.size() - 2);

    if (!token.empty() && token.front() == '"')
        token.erase(token.begin());
    if (!token.empty() && token.back() == '"')
        token.pop_back();

    return trim_ascii(token);
}

bool parse_game_line(const std::string& line, GameInput& game) {
    game.startFen.clear();
    game.moves.clear();

    std::string work = line;
    auto comment = work.find('#');
    if (comment != std::string::npos)
        work.resize(comment);

    const auto split = work.find('|');
    if (split != std::string::npos)
    {
        game.startFen = normalize_token(work.substr(0, split));
        work = work.substr(split + 1);
    }

    std::string current;
    std::istringstream iss(work);
    while (std::getline(iss, current, ','))
    {
        auto token = normalize_token(current);
        if (!token.empty())
            game.moves.push_back(token);
    }

    return !game.startFen.empty() || !game.moves.empty();
}

bool read_games_chunk(std::ifstream&       in,
                      std::vector<GameInput>& games,
                      std::size_t          maxGames,
                      std::size_t&         lineNo,
                      std::string&         err) {
    games.clear();
    if (maxGames == 0)
        maxGames = 1;

    auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
    std::string line;

    while (games.size() < maxGames && std::getline(in, line))
    {
        ++lineNo;

        auto first = std::find_if_not(line.begin(), line.end(), is_space);
        if (first == line.end())
            continue;

        GameInput game;
        if (!parse_game_line(line, game))
        {
            err = "Line " + std::to_string(lineNo) + ": no moves/FEN found";
            return false;
        }

        games.push_back(std::move(game));
    }

    return true;
}

bool is_integer_token(const std::string& s) {
    if (s.empty())
        return false;
    const std::size_t start = (s.front() == '-' || s.front() == '+') ? 1 : 0;
    if (start >= s.size())
        return false;
    return std::all_of(s.begin() + start, s.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
    });
}

bool parse_fen_or_epd_line(const std::string& line, std::string& fen) {
    fen.clear();
    std::string work = line;
    auto comment = work.find('#');
    if (comment != std::string::npos)
        work.resize(comment);
    work = trim_ascii(work);
    if (work.empty())
        return false;

    std::istringstream iss(work);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token)
        tokens.push_back(token);

    if (tokens.size() < 4)
        return false;

    std::string halfmove = "0";
    std::string fullmove = "1";
    if (tokens.size() >= 6 && is_integer_token(tokens[4]) && is_integer_token(tokens[5]))
    {
        halfmove = tokens[4];
        fullmove = tokens[5];
    }

    fen = tokens[0] + " " + tokens[1] + " " + tokens[2] + " " + tokens[3] + " " + halfmove
        + " " + fullmove;
    return true;
}

bool parse_decimal_int(const std::string& s, int& out) {
    if (s.empty())
        return false;

    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0')
        return false;
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        return false;

    out = static_cast<int>(value);
    return true;
}

bool compute_root_ply_from_fen(const std::string& fen, int& rootPly) {
    std::istringstream iss(fen);
    std::string        board;
    std::string        sideToMove;
    std::string        castling;
    std::string        enPassant;
    std::string        halfmoveClock;
    std::string        fullmoveNumber;
    if (!(iss >> board >> sideToMove >> castling >> enPassant >> halfmoveClock >> fullmoveNumber))
        return false;

    if (sideToMove != "w" && sideToMove != "b")
        return false;

    int fullmove = 0;
    if (!parse_decimal_int(fullmoveNumber, fullmove))
        return false;
    if (fullmove < 1)
        return false;

    const long long ply = 2LL * (fullmove - 1) + (sideToMove == "b" ? 1LL : 0LL);
    if (ply < 0 || ply > std::numeric_limits<int>::max())
        return false;

    rootPly = static_cast<int>(ply);
    return true;
}

bool read_fen_positions(const std::string& path, std::vector<GameInput>& games, std::string& err) {
    std::ifstream in(path);
    games.clear();

    if (!in)
    {
        err = "Failed to open file";
        return false;
    }

    auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
    std::string line;
    std::size_t line_no = 0;

    while (std::getline(in, line))
    {
        ++line_no;
        auto first = std::find_if_not(line.begin(), line.end(), is_space);
        if (first == line.end() || *first == '#')
            continue;

        GameInput game;
        if (!parse_fen_or_epd_line(line, game.startFen))
        {
            err = "Line " + std::to_string(line_no) + ": invalid FEN/EPD line";
            return false;
        }

        games.push_back(std::move(game));
    }

    return true;
}

constexpr std::uint64_t FnvOffsetBasis64A = 14695981039346656037ULL;
constexpr std::uint64_t FnvOffsetBasis64B = 7809847782465536322ULL;
constexpr std::uint64_t FnvPrime64        = 1099511628211ULL;

std::uint64_t hash_fen64(std::string_view fen, std::uint64_t seed) {
    std::uint64_t h = seed;
    for (unsigned char c : fen)
    {
        h ^= c;
        h *= FnvPrime64;
    }
    return h;
}

struct Hash128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    bool operator==(const Hash128& other) const {
        return lo == other.lo && hi == other.hi;
    }
};

struct Hash128Hasher {
    std::size_t operator()(const Hash128& v) const {
        return static_cast<std::size_t>(v.lo ^ (v.hi * 0x9e3779b97f4a7c15ULL));
    }
};

struct NnueFenKeyView {
    std::string_view board;
    std::string_view sideToMove;
};

bool extract_fen_field(std::string_view row, std::string_view& fen) {
    if (row.size() < 4 || row.front() != '"')
        return false;

    const std::size_t end = row.find("\",");
    if (end == std::string_view::npos || end < 2)
        return false;

    fen = row.substr(1, end - 1);
    return true;
}

bool extract_nnue_fen_key(std::string_view fen, NnueFenKeyView& key) {
    const std::size_t space1 = fen.find(' ');
    if (space1 == std::string_view::npos || space1 == 0)
        return false;

    const std::size_t sideStart = space1 + 1;
    if (sideStart >= fen.size())
        return false;

    const std::size_t space2 = fen.find(' ', sideStart);
    if (space2 == sideStart)
        return false;

    key.board = fen.substr(0, space1);
    key.sideToMove = space2 == std::string_view::npos ? fen.substr(sideStart)
                                                       : fen.substr(sideStart, space2 - sideStart);
    if (key.sideToMove.empty())
        return false;
    return true;
}

std::uint64_t hash_nnue_key64(const NnueFenKeyView& key, std::uint64_t seed) {
    std::uint64_t h = hash_fen64(key.board, seed);
    h ^= ' ';
    h *= FnvPrime64;
    h = hash_fen64(key.sideToMove, h);
    return h;
}

Hash128 hash_nnue_key128(const NnueFenKeyView& key) {
    return Hash128{hash_nnue_key64(key, FnvOffsetBasis64A),
                   hash_nnue_key64(key, FnvOffsetBasis64B)};
}

template<typename IsUniqueFn>
bool write_text_with_optional_dedup(const std::string& text,
                                    std::ostream&      out,
                                    std::size_t        maxRows,
                                    bool               dedup,
                                    IsUniqueFn&&       isUnique,
                                    std::size_t&       writtenRows,
                                    std::string&       error) {
    writtenRows = 0;
    std::size_t offset = 0;

    while (offset < text.size() && writtenRows < maxRows)
    {
        const std::size_t newline = text.find('\n', offset);
        const bool        hasNewline = newline != std::string::npos;
        const std::size_t lineEnd = hasNewline ? newline : text.size();
        const std::size_t rowSize = lineEnd - offset;

        std::string_view row(text.data() + offset, rowSize);

        bool keep = true;
        if (dedup)
        {
            std::string_view fen;
            if (!extract_fen_field(row, fen))
            {
                error = "Failed to parse generated CSV row during dedup";
                return false;
            }
            NnueFenKeyView key;
            if (!extract_nnue_fen_key(fen, key))
            {
                error = "Failed to parse FEN key during dedup";
                return false;
            }
            keep = isUnique(key);
        }

        if (keep)
        {
            out.write(text.data() + offset, static_cast<std::streamsize>(rowSize));
            out.put('\n');
            writtenRows++;
        }

        if (!hasNewline)
            break;
        offset = newline + 1;
    }

    return true;
}

std::size_t offset_after_n_lines(const std::string& text, std::size_t lines) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < lines && offset < text.size(); ++i)
    {
        const std::size_t end = text.find('\n', offset);
        if (end == std::string::npos)
            return text.size();
        offset = end + 1;
    }
    return offset;
}

void release_game_storage(GameInput& game) {
    std::vector<std::string>().swap(game.moves);
    std::string().swap(game.startFen);
}

bool write_count_cache_file(const std::string& outPath, std::size_t rows) {
    if (outPath.empty())
        return true;

    std::ofstream countFile(outPath + ".count", std::ios::binary | std::ios::trunc);
    if (!countFile)
        return false;

    countFile << rows;
    return bool(countFile);
}

bool read_count_cache_file(const std::string& outPath, std::size_t& rows) {
    rows = 0;
    if (outPath.empty())
        return false;

    std::ifstream countFile(outPath + ".count", std::ios::binary);
    if (!countFile)
        return false;

    std::string text;
    countFile >> text;
    if (!countFile || text.empty())
        return false;

    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;

    rows = static_cast<std::size_t>(value);
    return true;
}

bool count_existing_csv_rows(const std::string& csvPath, std::size_t& rows) {
    rows = 0;
    std::ifstream in(csvPath, std::ios::binary);
    if (!in)
        return false;

    std::string line;
    std::size_t nonEmptyLines = 0;
    while (std::getline(in, line))
    {
        const bool nonEmpty =
          std::any_of(line.begin(), line.end(), [](char c) {
              return !std::isspace(static_cast<unsigned char>(c));
          });
        if (nonEmpty)
            nonEmptyLines++;
    }

    rows = nonEmptyLines > 0 ? nonEmptyLines - 1 : 0;
    return true;
}

bool write_u32_le(std::ostream& out, std::uint32_t v) {
    const unsigned char b[4] = {
      static_cast<unsigned char>(v & 0xFFu),
      static_cast<unsigned char>((v >> 8) & 0xFFu),
      static_cast<unsigned char>((v >> 16) & 0xFFu),
      static_cast<unsigned char>((v >> 24) & 0xFFu),
    };
    out.write(reinterpret_cast<const char*>(b), 4);
    return bool(out);
}

bool write_u64_le(std::ostream& out, std::uint64_t v) {
    const unsigned char b[8] = {
      static_cast<unsigned char>(v & 0xFFu),
      static_cast<unsigned char>((v >> 8) & 0xFFu),
      static_cast<unsigned char>((v >> 16) & 0xFFu),
      static_cast<unsigned char>((v >> 24) & 0xFFu),
      static_cast<unsigned char>((v >> 32) & 0xFFu),
      static_cast<unsigned char>((v >> 40) & 0xFFu),
      static_cast<unsigned char>((v >> 48) & 0xFFu),
      static_cast<unsigned char>((v >> 56) & 0xFFu),
    };
    out.write(reinterpret_cast<const char*>(b), 8);
    return bool(out);
}

bool read_u32_le(std::istream& in, std::uint32_t& v) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in)
        return false;
    v = (std::uint32_t(b[0]) << 0) | (std::uint32_t(b[1]) << 8) | (std::uint32_t(b[2]) << 16)
      | (std::uint32_t(b[3]) << 24);
    return true;
}

bool read_u64_le(std::istream& in, std::uint64_t& v) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in)
        return false;
    v = (std::uint64_t(b[0]) << 0) | (std::uint64_t(b[1]) << 8) | (std::uint64_t(b[2]) << 16)
      | (std::uint64_t(b[3]) << 24) | (std::uint64_t(b[4]) << 32) | (std::uint64_t(b[5]) << 40)
      | (std::uint64_t(b[6]) << 48) | (std::uint64_t(b[7]) << 56);
    return true;
}

bool load_dedup_cache(const std::string&                     path,
                      std::size_t                            hashBits,
                      std::unordered_set<std::uint64_t>&     seen64,
                      std::unordered_set<Hash128, Hash128Hasher>& seen128,
                      std::string&                           err,
                      std::size_t&                           loadedCount) {
    loadedCount = 0;
    if (path.empty())
        return true;

    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(path, ec))
        return true;
    if (ec)
    {
        err = "Failed to stat dedup cache file: " + path;
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "Failed to open dedup cache file: " + path;
        return false;
    }

    char magic[8];
    in.read(magic, 8);
    if (!in || std::string_view(magic, 8) != std::string_view("NNDEDUP1", 8))
    {
        err = "Invalid dedup cache header in: " + path;
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t fileHashBits = 0;
    std::uint64_t entries = 0;
    if (!read_u32_le(in, version) || !read_u32_le(in, fileHashBits) || !read_u64_le(in, entries))
    {
        err = "Corrupt dedup cache header in: " + path;
        return false;
    }
    if (version != 1)
    {
        err = "Unsupported dedup cache version in: " + path;
        return false;
    }
    if (fileHashBits != hashBits)
    {
        err = "Dedup cache hash-bits mismatch in: " + path;
        return false;
    }

    for (std::uint64_t i = 0; i < entries; ++i)
    {
        if (hashBits == 64)
        {
            std::uint64_t v = 0;
            if (!read_u64_le(in, v))
            {
                err = "Corrupt dedup cache payload in: " + path;
                return false;
            }
            seen64.insert(v);
        }
        else
        {
            std::uint64_t lo = 0;
            std::uint64_t hi = 0;
            if (!read_u64_le(in, lo) || !read_u64_le(in, hi))
            {
                err = "Corrupt dedup cache payload in: " + path;
                return false;
            }
            seen128.insert(Hash128{lo, hi});
        }
    }

    loadedCount = hashBits == 64 ? seen64.size() : seen128.size();
    return true;
}

bool save_dedup_cache(const std::string&                     path,
                      std::size_t                            hashBits,
                      const std::unordered_set<std::uint64_t>& seen64,
                      const std::unordered_set<Hash128, Hash128Hasher>& seen128,
                      std::string&                           err) {
    if (path.empty())
        return true;

    namespace fs = std::filesystem;
    const fs::path cachePath(path);
    const fs::path parent = cachePath.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec)
        {
            err = "Failed to create dedup cache directory: " + parent.string();
            return false;
        }
    }

    const fs::path tmpPath = cachePath.string() + ".tmp";
    std::ofstream   out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "Failed to open temporary dedup cache file: " + tmpPath.string();
        return false;
    }

    out.write("NNDEDUP1", 8);
    if (!write_u32_le(out, 1u) || !write_u32_le(out, static_cast<std::uint32_t>(hashBits)))
    {
        err = "Failed to write dedup cache header: " + tmpPath.string();
        return false;
    }

    const std::uint64_t entries =
      hashBits == 64 ? static_cast<std::uint64_t>(seen64.size())
                     : static_cast<std::uint64_t>(seen128.size());
    if (!write_u64_le(out, entries))
    {
        err = "Failed to write dedup cache header: " + tmpPath.string();
        return false;
    }

    if (hashBits == 64)
    {
        for (std::uint64_t v : seen64)
        {
            if (!write_u64_le(out, v))
            {
                err = "Failed to write dedup cache payload: " + tmpPath.string();
                return false;
            }
        }
    }
    else
    {
        for (const Hash128& v : seen128)
        {
            if (!write_u64_le(out, v.lo) || !write_u64_le(out, v.hi))
            {
                err = "Failed to write dedup cache payload: " + tmpPath.string();
                return false;
            }
        }
    }

    out.flush();
    if (!out)
    {
        err = "Failed to finalize dedup cache file: " + tmpPath.string();
        return false;
    }
    out.close();
    if (!out)
    {
        err = "Failed to close dedup cache file: " + tmpPath.string();
        return false;
    }

    std::error_code ec;
    fs::remove(cachePath, ec);
    ec.clear();
    fs::rename(tmpPath, cachePath, ec);
    if (ec)
    {
        err = "Failed to move dedup cache into place: " + path;
        return false;
    }

    return true;
}

struct GameResult {
    std::string text;
    std::size_t positions = 0;
    bool        ready     = false;
};

void write_row(std::ostream& out,
               const Stockfish::Position& pos,
               Stockfish::Value nnue,
               Stockfish::Value psqt,
               Stockfish::Value positional) {
    const std::string fen = pos.fen();
    out << "\"" << fen << "\"," << int(psqt) << "," << int(positional) << "," << int(nnue)
        << "\n";
}

nnue_eval_shared::EvalOptions make_eval_options(const Options& opt) {
    nnue_eval_shared::EvalOptions shared;
    shared.outputNet = opt.outputNet == Options::OutputNet::Small
                     ? nnue_eval_shared::OutputNet::Small
                     : nnue_eval_shared::OutputNet::Big;
    shared.unscaledOutput = opt.unscaledOutput;
    shared.nnueLabelMode = opt.nnueLabelMode;
    return shared;
}

bool evaluate_game(int                                   gameId,
                   const GameInput&                      game,
                   const Options&                        opt,
                   Stockfish::Eval::NNUE::Networks&      networks,
                   Stockfish::Eval::NNUE::AccumulatorCaches& caches,
                   Stockfish::Eval::NNUE::AccumulatorStack&  accumulators,
                   std::string&                          text,
                   std::size_t&                          positions,
                   std::string&                          error) {
    using namespace Stockfish;
    namespace NN = Eval::NNUE;

    const std::string& startFen = game.startFen.empty() ? StartFEN : game.startFen;
    const auto&        moves = game.moves;
    int                rootPly = 0;
    if (!game.startFen.empty())
    {
        if (!compute_root_ply_from_fen(startFen, rootPly))
        {
            error = "Invalid start FEN at game " + std::to_string(gameId) + ": " + startFen;
            return false;
        }
    }

    Position               pos;
    std::vector<StateInfo> states(moves.size() + 1);
    pos.set(startFen, false, &states[0]);
    accumulators.reset();

    std::ostringstream out;
    positions = 0;

    const auto sharedOpt = make_eval_options(opt);
    const bool hasToPlyLimit = opt.toPly > 0;

    if (opt.fromPly <= rootPly && (!hasToPlyLimit || rootPly <= opt.toPly))
    {
        const auto evaluated =
          nnue_eval_shared::evaluate_position(sharedOpt, pos, networks, caches, accumulators);
        write_row(out, pos, evaluated.nnue, evaluated.psqt, evaluated.positional);
        positions++;
    }

    int accumPly = 0;
    int currentPly = rootPly;
    for (size_t i = 0; i < moves.size(); ++i)
    {
        const std::string& moveStr = moves[i];
        Move               m       = UCIEngine::to_move(pos, moveStr);
        if (m == Move::none())
        {
            error = "Illegal move at game " + std::to_string(gameId) + ", ply "
                  + std::to_string(i + 1) + ": " + moveStr;
            return false;
        }

        if (accumPly + 1 >= int(Stockfish::Eval::NNUE::AccumulatorStack::MaxSize))
        {
            accumulators.reset();
            accumPly = 0;
        }

        auto [dirtyPiece, dirtyThreats] = accumulators.push();
        pos.do_move(m, states[i + 1], pos.gives_check(m), dirtyPiece, dirtyThreats);
        accumPly++;
        currentPly++;

        if (hasToPlyLimit && currentPly > opt.toPly)
            break;

        if (currentPly < opt.fromPly)
            continue;

        const auto evaluated =
          nnue_eval_shared::evaluate_position(sharedOpt, pos, networks, caches, accumulators);
        write_row(out, pos, evaluated.nnue, evaluated.psqt, evaluated.positional);
        positions++;
    }

    text = out.str();
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        print_usage();
        return 2;
    }

    const bool hasMovesInput = !opt.movesPaths.empty();

    std::vector<GameInput> fenGames;
    std::string            readErr;
    if (!hasMovesInput)
    {
        if (!read_fen_positions(opt.fensPath, fenGames, readErr))
        {
            std::cerr << "Failed to read FEN file: " << opt.fensPath << "\n";
            std::cerr << "Error: " << readErr << "\n";
            return 1;
        }
        if (fenGames.empty())
        {
            std::cerr << "No FEN positions found in file: " << opt.fensPath << "\n";
            return 1;
        }
    }

    using namespace Stockfish;
    namespace NN = Eval::NNUE;

    Bitboards::init();
    Position::init();

    auto networks = std::make_unique<NN::Networks>(
      NN::EvalFile{EvalFileDefaultNameBig, "None", ""},
      NN::EvalFile{EvalFileDefaultNameSmall, "None", ""});

    const std::string binaryDir = CommandLine::get_binary_directory(argv[0]);
    const std::string bigPath =
      resolve_eval_file_path(opt.bigEvalFile, binaryDir, EvalFileDefaultNameBig);
    const std::string smallPath =
      resolve_eval_file_path(opt.smallEvalFile, binaryDir, EvalFileDefaultNameSmall);

    if (!check_eval_file(bigPath, "Big NNUE") || !check_eval_file(smallPath, "Small NNUE"))
        return 1;

    networks->big.load(binaryDir, bigPath);
    networks->small.load(binaryDir, smallPath);
    auto onVerify = [](std::string_view msg) { std::cerr << msg; };
    networks->big.verify(bigPath, onVerify);
    networks->small.verify(smallPath, onVerify);

    std::ofstream outFile;
    std::ostream* out = &std::cout;
    bool          headerWritten = false;
    std::size_t   existingRowsBase = 0;

    auto open_output = [&]() -> bool {
        if (opt.outPath.empty())
        {
            if (!headerWritten)
            {
                *out << "fen,psqt,positional,nnue\n";
                headerWritten = true;
            }
            return true;
        }

        namespace fs = std::filesystem;
        bool fileExists = false;
        bool fileHasContent = false;
        if (opt.appendOut)
        {
            std::error_code ec;
            fileExists = fs::exists(opt.outPath, ec);
            if (ec)
            {
                std::cerr << "Failed to stat output file: " << opt.outPath << "\n";
                return false;
            }
            if (fileExists)
            {
                const auto size = fs::file_size(opt.outPath, ec);
                if (ec)
                {
                    std::cerr << "Failed to stat output file size: " << opt.outPath << "\n";
                    return false;
                }
                fileHasContent = size > 0;
            }
        }

        const std::ios::openmode mode = opt.appendOut
                                       ? (std::ios::binary | std::ios::app)
                                       : (std::ios::binary | std::ios::trunc);
        outFile.open(opt.outPath, mode);
        if (!outFile)
        {
            std::cerr << "Failed to open output file: " << opt.outPath << "\n";
            return false;
        }
        out = &outFile;

        if (!opt.appendOut || !fileHasContent)
            *out << "fen,psqt,positional,nnue\n";
        else
        {
            if (!read_count_cache_file(opt.outPath, existingRowsBase))
            {
                if (!count_existing_csv_rows(opt.outPath, existingRowsBase))
                {
                    existingRowsBase = 0;
                    std::cerr << "Warning: failed to read " << opt.outPath
                              << ".count and failed to scan existing CSV for append mode; "
                                 "row count cache will be rebuilt from this run.\n";
                }
                else
                {
                    std::cerr << "Warning: missing/invalid " << opt.outPath
                              << ".count; scanned existing CSV to recover row count: "
                              << existingRowsBase << "\n";
                }
            }
        }
        headerWritten = true;
        return true;
    };

    if (!open_output())
        return 1;

    std::unordered_set<std::uint64_t> seen64;
    std::unordered_set<Hash128, Hash128Hasher> seen128;
    if (opt.dedup && opt.dedupReserve && opt.dedupReserveCount > 0)
    {
        if (opt.hashBits == 64)
            seen64.reserve(opt.dedupReserveCount);
        else
            seen128.reserve(opt.dedupReserveCount);
    }

    if (opt.dedup && !opt.dedupCachePath.empty())
    {
        std::string loadErr;
        std::size_t loadedCount = 0;
        if (!load_dedup_cache(opt.dedupCachePath, opt.hashBits, seen64, seen128, loadErr, loadedCount))
        {
            std::cerr << loadErr << "\n";
            return 1;
        }
    }

    auto is_unique_fen = [&](const NnueFenKeyView& key) -> bool {
        if (opt.hashBits == 64)
            return seen64.insert(hash_nnue_key64(key, FnvOffsetBasis64A)).second;

        return seen128.insert(hash_nnue_key128(key)).second;
    };

    std::size_t nextFlushMark = (opt.outPath.empty() || opt.flushEvery == 0) ? 0 : opt.flushEvery;
    bool        countCacheWarned = false;
    auto maintain_output_files = [&](std::size_t writtenRows, bool force) {
        if (opt.outPath.empty())
            return;

        bool shouldFlush = force;
        if (!force && nextFlushMark > 0 && writtenRows >= nextFlushMark)
        {
            shouldFlush = true;
            while (nextFlushMark > 0 && writtenRows >= nextFlushMark)
            {
                if (nextFlushMark > std::numeric_limits<std::size_t>::max() - opt.flushEvery)
                {
                    nextFlushMark = 0;
                    break;
                }
                nextFlushMark += opt.flushEvery;
            }
        }

        if (!shouldFlush)
            return;

        out->flush();
        if (!write_count_cache_file(opt.outPath, existingRowsBase + writtenRows) && !countCacheWarned)
        {
            std::cerr << "Warning: failed to write row-count cache file: " << opt.outPath
                      << ".count\n";
            countCacheWarned = true;
        }
    };

    const auto progressStart = std::chrono::steady_clock::now();
    std::size_t nextProgressMark = opt.progressEvery > 0 ? opt.progressEvery : 0;
    auto maybe_print_progress =
      [&](std::size_t writtenRows, std::size_t gamesDone, std::size_t totalGamesExpected) {
          if (opt.progressEvery == 0 || nextProgressMark == 0)
              return;

          while (writtenRows >= nextProgressMark)
          {
              const auto now = std::chrono::steady_clock::now();
              const auto elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - progressStart)
                  .count();
              const double elapsedSec = elapsedMs > 0 ? elapsedMs / 1000.0 : 0.0;

              std::cerr << "[progress] rows=" << writtenRows;
              if (opt.maxPositions > 0)
              {
                  const double pct = 100.0 * double(writtenRows) / double(opt.maxPositions);
                  std::cerr << "/" << opt.maxPositions << " (" << pct << "%)";
              }
              std::cerr << ", games=" << gamesDone;
              if (totalGamesExpected > 0)
                  std::cerr << "/" << totalGamesExpected;
              if (elapsedSec > 0.0)
                  std::cerr << ", rows/s=" << std::size_t(double(writtenRows) / elapsedSec);
              std::cerr << "\n";

              if (nextProgressMark > std::numeric_limits<std::size_t>::max() - opt.progressEvery)
              {
                  nextProgressMark = 0;
                  break;
              }
              nextProgressMark += opt.progressEvery;
          }
      };

    const std::size_t threadCount = std::max<std::size_t>(1, opt.threads);
    const std::size_t batchSize = std::max<std::size_t>(threadCount * 512, 4096);
    auto should_stop_writing = [&]() { return opt.maxPositions > 0; };

    std::unique_ptr<NN::AccumulatorCaches> singleThreadCaches;
    std::unique_ptr<NN::AccumulatorStack>  singleThreadAccumulators;
    if (threadCount == 1)
    {
        singleThreadCaches = std::make_unique<NN::AccumulatorCaches>(*networks);
        singleThreadAccumulators = std::make_unique<NN::AccumulatorStack>();
    }

    std::size_t totalPositions = 0;
    std::size_t gamesDone = 0;

    auto process_games_batch =
      [&](std::vector<GameInput>& gamesBatch,
          std::size_t             gameIdBase,
          std::size_t             totalGamesExpected,
          std::string&            errorOut) -> bool {
          if (gamesBatch.empty())
              return true;

          if (threadCount == 1)
          {
              for (std::size_t g = 0; g < gamesBatch.size(); ++g)
              {
                  if (should_stop_writing() && totalPositions >= opt.maxPositions)
                      break;

                  std::string text;
                  std::size_t positions = 0;
                  std::string error;
                  const int   gameId = int(gameIdBase + g + 1);

                  if (!evaluate_game(gameId,
                                     gamesBatch[g],
                                     opt,
                                     *networks,
                                     *singleThreadCaches,
                                     *singleThreadAccumulators,
                                     text,
                                     positions,
                                     error))
                  {
                      errorOut = std::move(error);
                      return false;
                  }

                  const std::size_t remaining =
                    should_stop_writing() ? opt.maxPositions - totalPositions
                                          : std::numeric_limits<std::size_t>::max();
                  std::size_t writtenPositions = 0;

                  if (!opt.dedup)
                  {
                      writtenPositions = std::min(positions, remaining);

                      if (writtenPositions > 0)
                      {
                          if (writtenPositions == positions)
                              *out << text;
                          else
                              *out << text.substr(0, offset_after_n_lines(text, writtenPositions));
                      }
                  }
                  else if (!write_text_with_optional_dedup(
                             text, *out, remaining, true, is_unique_fen, writtenPositions, error))
                  {
                      errorOut = std::move(error);
                      return false;
                  }

                  if (writtenPositions > 0)
                      totalPositions += writtenPositions;
                  gamesDone++;
                  maybe_print_progress(totalPositions, gamesDone, totalGamesExpected);
                  maintain_output_files(totalPositions, false);
                  release_game_storage(gamesBatch[g]);

                  if (should_stop_writing() && totalPositions >= opt.maxPositions)
                      break;
              }

              return true;
          }

          std::atomic<std::size_t> nextIdx{0};
          std::atomic<bool>        failed{false};
          std::atomic<bool>        stopRequested{false};
          std::string              workerError;
          std::mutex               errorMutex;

          std::vector<GameResult> results(gamesBatch.size());
          std::mutex              resultsMutex;
          std::condition_variable resultsCv;

          auto worker = [&]() {
              auto caches = std::make_unique<NN::AccumulatorCaches>(*networks);
              auto accumulators = std::make_unique<NN::AccumulatorStack>();

              while (true)
              {
                  if (failed.load() || stopRequested.load())
                      break;

                  std::size_t idx = nextIdx.fetch_add(1);
                  if (idx >= gamesBatch.size())
                      break;

                  std::string text;
                  std::size_t positions = 0;
                  std::string error;
                  const int   gameId = int(gameIdBase + idx + 1);
                  bool        ok =
                    evaluate_game(gameId, gamesBatch[idx], opt, *networks, *caches, *accumulators,
                                  text, positions, error);

                  {
                      std::lock_guard<std::mutex> resultsLock(resultsMutex);
                      results[idx].text = std::move(text);
                      results[idx].positions = positions;
                      results[idx].ready = true;
                  }

                  if (!ok)
                  {
                      {
                          std::lock_guard<std::mutex> errorLock(errorMutex);
                          if (!failed.load())
                              workerError = std::move(error);
                      }
                      failed.store(true);
                  }

                  resultsCv.notify_all();

                  if (!ok)
                      break;
              }
          };

          std::vector<std::thread> threads;
          threads.reserve(threadCount);
          for (std::size_t i = 0; i < threadCount; ++i)
              threads.emplace_back(worker);

          std::size_t nextToWrite = 0;
          while (nextToWrite < gamesBatch.size())
          {
              if (should_stop_writing() && totalPositions >= opt.maxPositions)
              {
                  stopRequested.store(true);
                  break;
              }

              std::unique_lock<std::mutex> resultsLock(resultsMutex);
              resultsCv.wait(resultsLock, [&]() {
                  return results[nextToWrite].ready || failed.load();
              });

              if (failed.load() && !results[nextToWrite].ready)
                  break;

              std::string text = std::move(results[nextToWrite].text);
              std::size_t positions = results[nextToWrite].positions;
              std::string().swap(results[nextToWrite].text);
              results[nextToWrite].ready = false;
              resultsLock.unlock();

              const std::size_t remaining =
                should_stop_writing() ? opt.maxPositions - totalPositions
                                      : std::numeric_limits<std::size_t>::max();
              std::size_t writtenPositions = 0;

              if (!opt.dedup)
              {
                  writtenPositions = std::min(positions, remaining);

                  if (writtenPositions > 0)
                  {
                      if (writtenPositions == positions)
                          *out << text;
                      else
                          *out << text.substr(0, offset_after_n_lines(text, writtenPositions));
                  }
              }
              else
              {
                  std::string writeError;
                  if (!write_text_with_optional_dedup(
                        text, *out, remaining, true, is_unique_fen, writtenPositions, writeError))
                  {
                      stopRequested.store(true);
                      {
                          std::lock_guard<std::mutex> errorLock(errorMutex);
                          if (!failed.load())
                              workerError = std::move(writeError);
                      }
                      failed.store(true);
                      break;
                  }
              }

              if (writtenPositions > 0)
                  totalPositions += writtenPositions;

              gamesDone++;
              maybe_print_progress(totalPositions, gamesDone, totalGamesExpected);
              maintain_output_files(totalPositions, false);
              release_game_storage(gamesBatch[nextToWrite]);

              if (should_stop_writing() && totalPositions >= opt.maxPositions)
              {
                  stopRequested.store(true);
                  break;
              }

              nextToWrite++;
          }

          for (auto& t : threads)
              t.join();

          if (failed.load())
          {
              std::lock_guard<std::mutex> errorLock(errorMutex);
              errorOut = workerError;
              return false;
          }

          return true;
      };

    bool anyGamesSeen = false;
    if (hasMovesInput)
    {
        std::size_t gameIdBase = 0;
        for (const auto& movesPath : opt.movesPaths)
        {
            std::ifstream in(movesPath);
            if (!in)
            {
                std::cerr << "Failed to read moves file: " << movesPath << "\n";
                std::cerr << "Error: Failed to open file\n";
                return 1;
            }

            std::size_t lineNo = 0;
            while (true)
            {
                std::vector<GameInput> gamesBatch;
                if (!read_games_chunk(in, gamesBatch, batchSize, lineNo, readErr))
                {
                    std::cerr << "Failed to read moves file: " << movesPath << "\n";
                    std::cerr << "Error: " << readErr << "\n";
                    return 1;
                }

                if (gamesBatch.empty())
                    break;

                anyGamesSeen = true;
                const std::size_t thisBatchBase = gameIdBase;
                gameIdBase += gamesBatch.size();

                std::string processError;
                if (!process_games_batch(
                      gamesBatch, thisBatchBase, opt.gamesTotalHint, processError))
                {
                    std::cerr << processError << "\n";
                    return 1;
                }

                if (should_stop_writing() && totalPositions >= opt.maxPositions)
                    break;
            }

            if (should_stop_writing() && totalPositions >= opt.maxPositions)
                break;
        }

        if (!anyGamesSeen)
        {
            std::cerr << "No games found in moves files\n";
            return 1;
        }
    }
    else
    {
        anyGamesSeen = true;
        std::string processError;
        if (!process_games_batch(fenGames, 0, fenGames.size(), processError))
        {
            std::cerr << processError << "\n";
            return 1;
        }
    }

    maintain_output_files(totalPositions, true);

    if (opt.dedup && !opt.dedupCachePath.empty())
    {
        std::string saveErr;
        if (!save_dedup_cache(opt.dedupCachePath, opt.hashBits, seen64, seen128, saveErr))
        {
            std::cerr << saveErr << "\n";
            return 1;
        }
    }

    return 0;
}

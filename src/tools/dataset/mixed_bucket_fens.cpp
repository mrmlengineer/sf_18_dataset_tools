/*
  Mixed bucketed FEN dataset builder.

  Input format (stdin or --in):
    F|uci1,uci2,...              -> include from ply 0 across all 16 buckets
    E|uci1,uci2,...              -> include from ply 0, endgame-only (b00/b01)
    F1|uci1,uci2,...             -> include from ply 1 across all 16 buckets
    E1|uci1,uci2,...             -> include from ply 1, endgame-only (b00/b01)
    F|<start_fen>|uci1,uci2,...  -> full game from a custom start position
    E|<start_fen>|uci1,uci2,...  -> endgame-only from a custom start position
    F|<start_fen>|               -> root-position-only record

  Output:
    - 16 CSV files (fen,psqt,positional,nnue), one per bucket
    - one manifest JSON
    - one .count JSON per bucket, containing total_rows + piece_count_counts
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "bitboard.h"
#include "engine.h"
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
constexpr std::size_t BucketCount = 16;
constexpr std::uint64_t FnvOffsetBasis64A = 14695981039346656037ULL;
constexpr std::uint64_t FnvPrime64        = 1099511628211ULL;
constexpr std::size_t DedupMaxLoadPercent = 70;
constexpr std::size_t DedupMinCapacity    = 16;

class FlatHashSet64 {
  public:
    bool insert(std::uint64_t value) {
        ensure_capacity_for_insert();
        return insert_into(slots_, occupied_, size_, value);
    }

    void reserve(std::size_t expectedSize) {
        const std::size_t requiredCapacity = capacity_for_size(expectedSize);
        if (requiredCapacity > slots_.size())
            rehash(requiredCapacity);
    }

    [[nodiscard]] std::size_t size() const { return size_; }

  private:
    std::vector<std::uint64_t> slots_;
    std::vector<std::uint64_t> occupied_;
    std::size_t                size_            = 0;
    std::size_t                resizeThreshold_ = 0;

    static std::uint64_t mix_hash(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    static std::size_t ceil_div(std::size_t a, std::size_t b) {
        return a / b + (a % b != 0 ? 1 : 0);
    }

    static std::size_t capacity_for_size(std::size_t size) {
        if (size == 0)
            return DedupMinCapacity;
        if (size > (std::numeric_limits<std::size_t>::max() - 1) / 100)
            return std::numeric_limits<std::size_t>::max();
        return std::max(DedupMinCapacity, ceil_div(size * 100 + 1, DedupMaxLoadPercent));
    }

    static std::size_t resize_threshold_for_capacity(std::size_t capacity) {
        return std::max<std::size_t>(1, (capacity * DedupMaxLoadPercent) / 100);
    }

    static bool bit_is_set(const std::vector<std::uint64_t>& occupied, std::size_t index) {
        return (occupied[index >> 6] >> (index & 63)) & 1ULL;
    }

    static void set_bit(std::vector<std::uint64_t>& occupied, std::size_t index) {
        occupied[index >> 6] |= (1ULL << (index & 63));
    }

    static bool insert_into(std::vector<std::uint64_t>& slots,
                            std::vector<std::uint64_t>& occupied,
                            std::size_t&                size,
                            std::uint64_t               value) {
        const std::size_t capacity = slots.size();
        std::size_t       index    = static_cast<std::size_t>(mix_hash(value) % capacity);
        while (true)
        {
            if (!bit_is_set(occupied, index))
            {
                slots[index] = value;
                set_bit(occupied, index);
                ++size;
                return true;
            }
            if (slots[index] == value)
                return false;
            ++index;
            if (index == capacity)
                index = 0;
        }
    }

    void ensure_capacity_for_insert() {
        if (slots_.empty())
        {
            rehash(DedupMinCapacity);
            return;
        }
        if (size_ + 1 > resizeThreshold_)
            rehash(std::max(slots_.size() * 2, capacity_for_size(size_ + 1)));
    }

    void rehash(std::size_t newCapacity) {
        newCapacity = std::max(newCapacity, capacity_for_size(size_));

        std::vector<std::uint64_t> newSlots(newCapacity);
        std::vector<std::uint64_t> newOccupied((newCapacity + 63) / 64, 0);
        std::size_t                newSize = 0;

        for (std::size_t index = 0; index < slots_.size(); ++index)
            if (bit_is_set(occupied_, index))
                insert_into(newSlots, newOccupied, newSize, slots_[index]);

        slots_.swap(newSlots);
        occupied_.swap(newOccupied);
        size_ = newSize;
        resizeThreshold_ = resize_threshold_for_capacity(slots_.size());
    }
};

struct Options {
    std::string inPath;
    std::string outDir;
    std::string bucketOutputDir;
    std::string outputStem;
    std::string evalFile;
    std::string bigEvalFile;
    std::string smallEvalFile;
    nnue_eval_shared::NnueLabelMode nnueLabelMode = nnue_eval_shared::NnueLabelMode::Adjusted;
    std::vector<std::string> excludeHashFiles;
    std::size_t threads            = 1;
    std::size_t batchGames         = 100;
    std::size_t minFens            = 0;
    std::string minTargetDisplayLabel = "positions";
    std::int64_t minTargetDisplayOffset = 0;
    std::size_t progressEveryGames = 20000;
    std::size_t progressEverySeconds = 0;
    std::size_t flushEveryFens     = 1000000;
    std::string phaseLabel         = "process";
    bool        dedup              = true;
    bool        appendOut          = false;
    bool        streamOut          = false;
    bool        splitFinal         = false;
    bool        finalizeExistingSplit = false;
    double      trainRatio         = 0.90;
    double      validationRatio    = 0.05;
    double      testRatio          = 0.05;
    std::size_t maxValidationRows  = 500000;
    std::size_t maxTestRows        = 500000;
    std::size_t splitShards        = 256;
    std::size_t splitWorkers       = 1;
};

struct GameInput {
    bool                     includeFullGame = false;
    int                      fromPly = 0;
    std::optional<int>       minElo;
    std::string              source;
    std::string              phase;
    std::string              startFen;
    std::vector<std::string> moves;
};

struct PhaseStats {
    std::size_t positionsScanned = 0;
    std::size_t positionsWritten = 0;
    std::size_t duplicateSkipped = 0;
};

struct RowData {
    std::string       fen;
    int               pieceCount = 0;
    Stockfish::Value  psqt = Stockfish::VALUE_ZERO;
    Stockfish::Value  positional = Stockfish::VALUE_ZERO;
    Stockfish::Value  nnue = Stockfish::VALUE_ZERO;
    std::optional<int> minElo;
    std::string       source;
};

struct GameProcessResult {
    std::array<std::vector<RowData>, BucketCount> rowsByBucket;
    std::string                                   phase;
    std::size_t                                   positionsScanned  = 0;
    std::size_t                                   positionsSelected = 0;
    std::size_t                                   duplicateSkipped  = 0;
    std::uint64_t                                 gameSetupUs       = 0;
    std::uint64_t                                 moveParseUs       = 0;
    std::uint64_t                                 moveApplyUs       = 0;
    std::uint64_t                                 fenSerializeUs    = 0;
    std::uint64_t                                 dedupUs           = 0;
    std::uint64_t                                 rowBuildUs        = 0;
    bool                                          invalidGame       = false;
    bool                                          emptyGame         = false;
    std::string                                   fatalError;
};

struct BatchStats {
    std::size_t gamesProcessed        = 0;
    std::size_t gamesFull             = 0;
    std::size_t gamesEndgameOnly      = 0;
    std::size_t gamesSkippedNoMoves   = 0;
    std::size_t gamesSkippedInvalid   = 0;
    std::size_t positionsScanned      = 0;
    std::size_t positionsSelected     = 0;
    std::size_t duplicateSkipped      = 0;
    std::uint64_t gameSetupUs         = 0;
    std::uint64_t moveParseUs         = 0;
    std::uint64_t moveApplyUs         = 0;
    std::uint64_t fenSerializeUs      = 0;
    std::uint64_t dedupUs             = 0;
    std::uint64_t rowBuildUs          = 0;
    std::uint64_t batchSetupUs        = 0;
    std::uint64_t batchProcessUs      = 0;
    std::map<std::string, PhaseStats> phaseStats;
    std::vector<std::pair<std::string, PhaseStats>> phaseSegments;
};

struct BatchResult {
    std::size_t                                   sequence = 0;
    std::array<std::vector<RowData>, BucketCount> rowsByBucket;
    BatchStats                                    stats;
    std::string                                   fatalError;
};

struct WorkerResources {
    std::unique_ptr<Stockfish::Eval::NNUE::AccumulatorCaches> caches;
    std::unique_ptr<Stockfish::Eval::NNUE::AccumulatorStack>  accumulators;
    bool                                                     initialized = false;
};

struct SharedDedupState {
    std::array<FlatHashSet64, BucketCount>* seen64 = nullptr;
    std::array<std::mutex, BucketCount>*    mutexes = nullptr;
    bool                                    enabled = false;
};

struct WorkItem {
    std::size_t sequence = 0;
    std::vector<GameInput> games;
};

struct GlobalStats {
    std::size_t gamesSelected          = 0;
    std::size_t gamesSelectedFull      = 0;
    std::size_t gamesSelectedEndgame   = 0;
    std::size_t gamesProcessed         = 0;
    std::size_t gamesFull              = 0;
    std::size_t gamesEndgameOnly       = 0;
    std::size_t gamesSkippedNoMoves    = 0;
    std::size_t gamesSkippedInvalid    = 0;
    std::size_t positionsScanned       = 0;
    std::size_t positionsSelected      = 0;
    std::size_t duplicateFensSkipped   = 0;
    std::size_t fensWritten            = 0;
    std::size_t batchesCompleted       = 0;
    std::size_t workItemsSubmitted     = 0;
    std::uint64_t batchSetupUs         = 0;
    std::uint64_t batchProcessUs       = 0;
    std::uint64_t readerReadUs         = 0;
    std::uint64_t readerPushWaitUs     = 0;
    std::uint64_t resultPopWaitUs      = 0;
    std::uint64_t resultIntegrateUs    = 0;
    std::uint64_t outputWriteUs        = 0;
    std::uint64_t flushUs              = 0;
    std::uint64_t gameSetupUs          = 0;
    std::uint64_t moveParseUs          = 0;
    std::uint64_t moveApplyUs          = 0;
    std::uint64_t fenSerializeUs       = 0;
    std::uint64_t dedupUs              = 0;
    std::uint64_t rowBuildUs           = 0;
    std::map<std::string, PhaseStats> phaseStats;
};

struct ProcessRuntimeProgress {
    std::atomic<std::size_t> positionsScanned{0};
    std::atomic<std::size_t> positionsWritten{0};
    std::atomic<std::size_t> duplicateDiscarded{0};
    std::atomic<std::size_t> endgameWritten{0};
};

template<typename T>
class BoundedQueue {
  public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notFull_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_)
            return false;
        queue_.push_back(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        notEmpty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        item = std::move(queue_.front());
        queue_.pop_front();
        notFull_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

  private:
    std::size_t              capacity_;
    std::deque<T>            queue_;
    std::mutex               mutex_;
    std::condition_variable  notEmpty_;
    std::condition_variable  notFull_;
    bool                     closed_ = false;
};

std::string trim_ascii(const std::string& s) {
    auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
    auto begin = std::find_if_not(s.begin(), s.end(), is_space);
    auto end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
    if (begin >= end)
        return {};
    return std::string(begin, end);
}

std::string format_grouped_u64(std::uint64_t value) {
    const std::string digits = std::to_string(value);
    std::string result;
    result.reserve(digits.size() + digits.size() / 3);
    const std::size_t firstGroup = digits.size() % 3 == 0 ? 3 : digits.size() % 3;
    result.append(digits, 0, firstGroup);
    for (std::size_t i = firstGroup; i < digits.size(); i += 3)
    {
        result.push_back(',');
        result.append(digits, i, 3);
    }
    return result;
}

void add_phase_stats(PhaseStats& dst, const PhaseStats& src) {
    dst.positionsScanned += src.positionsScanned;
    dst.positionsWritten += src.positionsWritten;
    dst.duplicateSkipped += src.duplicateSkipped;
}

void print_phase_summary(const std::string& phaseName, const PhaseStats& stats, double elapsed) {
    const std::size_t discarded = stats.positionsScanned > stats.positionsWritten
                                  ? stats.positionsScanned - stats.positionsWritten
                                  : 0;
    const double rate = elapsed > 0.0 ? static_cast<double>(stats.positionsScanned) / elapsed : 0.0;
    std::cout << "[summary " << phaseName << "]"
              << " positions_scanned " << format_grouped_u64(stats.positionsScanned)
              << " | positions_written " << format_grouped_u64(stats.positionsWritten)
              << " | positions_discarded " << format_grouped_u64(discarded)
              << " | duplicated_discarded " << format_grouped_u64(stats.duplicateSkipped)
              << " | rate " << format_grouped_u64(static_cast<std::uint64_t>(std::llround(rate))) << " positions/s"
              << " | elapsed " << std::fixed << std::setprecision(1) << elapsed << "s\n";
    std::cout << "[phase " << phaseName << "] end\n";
    std::cout.flush();
}

std::string output_csv_header(const Options&) {
    return "fen,psqt,positional,nnue,min_elo,source\n";
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

bool parse_nonnegative_int(const std::string& text, int& value) {
    if (text.empty())
        return false;
    for (char c : text)
    {
        if (!std::isdigit(static_cast<unsigned char>(c)))
            return false;
    }

    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;
    if (parsed < 0 || parsed > std::numeric_limits<int>::max())
        return false;

    value = static_cast<int>(parsed);
    return true;
}

void print_usage() {
    std::cerr
      << "Usage:\n"
      << "  mixed_bucket_fens [--in <records.txt>] --out-dir <dir> --output-stem <name>\n"
      << "                    [--bucket-output-dir <dir>]\n"
      << "                    [--threads <N>] [--batch-games <N>] [--min-fens <N>]\n"
      << "                    [--min-target-display-label <label>] [--min-target-display-offset <N>]\n"
      << "                    [--progress-every-games <N>] [--progress-every-seconds <N>] [--phase-label <label>] [--flush-every-fens <N>]\n"
      << "                    [--split-final] [--finalize-existing-split]\n"
      << "                    [--train-ratio <f>] [--validation-ratio <f>] [--test-ratio <f>]\n"
      << "                    [--max-validation-rows <N>] [--max-test-rows <N>] [--split-shards <N>] [--split-workers <N>]\n"
      << "                    [--big <evalfile>] [--small <evalfile>]\n"
      << "                    [--nnue-label <adjusted|raw>]\n"
      << "                    [--exclude-hash-file <path>]... [--append-out] [--stream-out]\n"
      << "                    [--dedup|--no-dedup]\n\n"
      << "Input records:\n"
      << "  F|uci1,uci2,...              full game across all buckets from ply 0\n"
      << "  F1;min_elo=2000;source=name|uci1,... full game with per-row metadata\n"
      << "  E|uci1,uci2,...              endgame-only (b00/b01) from ply 0\n"
      << "  F1|uci1,uci2,...             full game across all buckets from ply 1\n"
      << "  E1|uci1,uci2,...             endgame-only (b00/b01) from ply 1\n"
      << "  F|<start_fen>|uci1,uci2,...  full game from custom start FEN\n"
      << "  E|<start_fen>|uci1,uci2,...  endgame-only from custom start FEN\n"
      << "  F|<start_fen>|               root-position-only record\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--in" && i + 1 < argc)
            opt.inPath = argv[++i];
        else if (arg == "--out-dir" && i + 1 < argc)
            opt.outDir = argv[++i];
        else if (arg == "--bucket-output-dir" && i + 1 < argc)
            opt.bucketOutputDir = argv[++i];
        else if (arg == "--output-stem" && i + 1 < argc)
            opt.outputStem = argv[++i];
        else if (arg == "--eval-file" && i + 1 < argc)
            opt.evalFile = argv[++i];
        else if (arg == "--big" && i + 1 < argc)
            opt.bigEvalFile = argv[++i];
        else if (arg == "--small" && i + 1 < argc)
            opt.smallEvalFile = argv[++i];
        else if (arg == "--exclude-hash-file" && i + 1 < argc)
            opt.excludeHashFiles.push_back(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)
            opt.threads = std::stoull(argv[++i]);
        else if (arg == "--batch-games" && i + 1 < argc)
            opt.batchGames = std::stoull(argv[++i]);
        else if (arg == "--min-fens" && i + 1 < argc)
            opt.minFens = std::stoull(argv[++i]);
        else if (arg == "--min-target-display-label" && i + 1 < argc)
            opt.minTargetDisplayLabel = argv[++i];
        else if (arg == "--min-target-display-offset" && i + 1 < argc)
            opt.minTargetDisplayOffset = std::stoll(argv[++i]);
        else if (arg == "--progress-every-games" && i + 1 < argc)
            opt.progressEveryGames = std::stoull(argv[++i]);
        else if (arg == "--progress-every-seconds" && i + 1 < argc)
            opt.progressEverySeconds = std::stoull(argv[++i]);
        else if (arg == "--phase-label" && i + 1 < argc)
            opt.phaseLabel = argv[++i];
        else if (arg == "--flush-every-fens" && i + 1 < argc)
            opt.flushEveryFens = std::stoull(argv[++i]);
        else if (arg == "--split-final")
            opt.splitFinal = true;
        else if (arg == "--finalize-existing-split")
            opt.finalizeExistingSplit = true;
        else if (arg == "--train-ratio" && i + 1 < argc)
            opt.trainRatio = std::stod(argv[++i]);
        else if (arg == "--validation-ratio" && i + 1 < argc)
            opt.validationRatio = std::stod(argv[++i]);
        else if (arg == "--test-ratio" && i + 1 < argc)
            opt.testRatio = std::stod(argv[++i]);
        else if (arg == "--max-validation-rows" && i + 1 < argc)
            opt.maxValidationRows = std::stoull(argv[++i]);
        else if (arg == "--max-test-rows" && i + 1 < argc)
            opt.maxTestRows = std::stoull(argv[++i]);
        else if (arg == "--split-shards" && i + 1 < argc)
            opt.splitShards = std::stoull(argv[++i]);
        else if (arg == "--split-workers" && i + 1 < argc)
            opt.splitWorkers = std::stoull(argv[++i]);
        else if (arg == "--output-scale" && i + 1 < argc)
            ++i;  // Deprecated compatibility flag: dataset heads are always emitted unscaled.
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
        else if (arg == "--dedup")
            opt.dedup = true;
        else if (arg == "--no-dedup")
            opt.dedup = false;
        else if (arg == "--append-out")
            opt.appendOut = true;
        else if (arg == "--stream-out")
            opt.streamOut = true;
        else
            return false;
    }

    if (opt.outDir.empty() || opt.outputStem.empty())
        return false;
    if (opt.threads == 0)
        opt.threads = 1;
    if (opt.batchGames == 0)
        opt.batchGames = 1;
    if (opt.splitFinal && opt.appendOut)
        return false;
    if (opt.streamOut && opt.appendOut)
        return false;
    if (opt.streamOut && opt.splitFinal)
        return false;
    if (opt.splitShards == 0)
        opt.splitShards = 1;
    if (opt.splitWorkers == 0)
        opt.splitWorkers = 1;
    return true;
}

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

bool parse_record_line(const std::string& line, GameInput& game, std::string& err) {
    game.includeFullGame = false;
    game.fromPly = 0;
    game.minElo.reset();
    game.source.clear();
    game.phase.clear();
    game.startFen.clear();
    game.moves.clear();

    std::string work = line;
    auto comment = work.find('#');
    if (comment != std::string::npos)
        work.resize(comment);
    work = trim_ascii(work);
    if (work.empty())
        return false;

    const auto split = work.find('|');
    if (split == std::string::npos || split == 0)
    {
        err = "missing mode separator '|'";
        return false;
    }

    const std::string modeWithMetadata = trim_ascii(work.substr(0, split));
    const auto metadataSplit = modeWithMetadata.find(';');
    const std::string mode = trim_ascii(
      metadataSplit == std::string::npos ? modeWithMetadata : modeWithMetadata.substr(0, metadataSplit));
    if (mode.empty())
    {
        err = "invalid mode '" + mode + "'";
        return false;
    }
    if (mode[0] == 'F')
        game.includeFullGame = true;
    else if (mode[0] == 'E')
        game.includeFullGame = false;
    else
    {
        err = "invalid mode '" + mode + "'";
        return false;
    }
    if (mode.size() > 1)
    {
        if (!parse_nonnegative_int(mode.substr(1), game.fromPly))
        {
            err = "invalid mode suffix '" + mode + "'";
            return false;
        }
    }
    if (metadataSplit != std::string::npos)
    {
        std::string metadataText = modeWithMetadata.substr(metadataSplit + 1);
        std::string metadataItem;
        std::istringstream metadataStream(metadataText);
        while (std::getline(metadataStream, metadataItem, ';'))
        {
            metadataItem = trim_ascii(metadataItem);
            if (metadataItem.empty())
                continue;
            const auto eq = metadataItem.find('=');
            if (eq == std::string::npos)
            {
                err = "invalid metadata item '" + metadataItem + "'";
                return false;
            }
            const std::string key = trim_ascii(metadataItem.substr(0, eq));
            const std::string value = trim_ascii(metadataItem.substr(eq + 1));
            if (key == "min_elo")
            {
                int parsedMinElo = 0;
                if (!parse_nonnegative_int(value, parsedMinElo))
                {
                    err = "invalid min_elo metadata '" + value + "'";
                    return false;
                }
                game.minElo = parsedMinElo;
            }
            else if (key == "source")
            {
                game.source = value;
            }
            else if (key == "phase")
            {
                game.phase = value;
            }
            else
            {
                err = "unknown metadata key '" + key + "'";
                return false;
            }
        }
    }

    std::string movesText = work.substr(split + 1);
    const auto fenSplit = movesText.find('|');
    if (fenSplit != std::string::npos)
    {
        game.startFen = normalize_token(movesText.substr(0, fenSplit));
        movesText = movesText.substr(fenSplit + 1);
        if (game.startFen.empty())
        {
            err = "empty start FEN";
            return false;
        }
    }

    std::string current;
    std::istringstream iss(movesText);
    while (std::getline(iss, current, ','))
    {
        auto token = normalize_token(current);
        if (!token.empty())
            game.moves.push_back(token);
    }

    // Collapse redundant standard-start FEN prefixes for move records, but keep
    // root-only records like startpos.epd valid.
    if (game.startFen == StartFEN && !game.moves.empty())
        game.startFen.clear();

    if (game.moves.empty() && game.startFen.empty())
    {
        err = "no moves found";
        return false;
    }

    return true;
}

bool should_select_position(const GameInput& game, int bucket) {
    return game.includeFullGame || bucket <= 1;
}

struct NnueFenKeyView {
    std::string_view board;
    std::string_view sideToMove;
};

int bucket_index(int pieceCount, Stockfish::Color stm);
bool extract_nnue_fen_key(std::string_view fen, NnueFenKeyView& key);
std::uint64_t hash_nnue_key64(const NnueFenKeyView& key, std::uint64_t seed);

nnue_eval_shared::EvalOptions make_eval_options(const Options& opt);

void append_current_position(const GameInput&                          game,
                             const Options&                            opt,
                             Stockfish::Position&                      pos,
                             Stockfish::Eval::NNUE::Networks&          networks,
                             Stockfish::Eval::NNUE::AccumulatorCaches& caches,
                             Stockfish::Eval::NNUE::AccumulatorStack&  accumulators,
                             SharedDedupState*                         dedupState,
                             int                                       currentPly,
                             GameProcessResult&                        result) {
    if (currentPly < game.fromPly)
        return;

    const int pieceCount = pos.count<Stockfish::ALL_PIECES>();
    const int bucket = bucket_index(pieceCount, pos.side_to_move());
    if (!should_select_position(game, bucket))
        return;

    result.positionsScanned++;
    const auto fenStartedAt = std::chrono::steady_clock::now();
    std::string fen = pos.fen();
    result.fenSerializeUs += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - fenStartedAt).count());
    if (dedupState && dedupState->enabled)
    {
        const auto dedupStartedAt = std::chrono::steady_clock::now();
        NnueFenKeyView key;
        if (!extract_nnue_fen_key(fen, key))
        {
            result.dedupUs += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - dedupStartedAt).count());
            result.fatalError = "Failed to parse generated FEN for dedup";
            return;
        }
        const std::uint64_t hash = hash_nnue_key64(key, FnvOffsetBasis64A);
        {
            std::lock_guard<std::mutex> lock((*dedupState->mutexes)[static_cast<std::size_t>(bucket)]);
            if (!(*dedupState->seen64)[static_cast<std::size_t>(bucket)].insert(hash))
            {
                result.dedupUs += static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - dedupStartedAt).count());
                result.duplicateSkipped++;
                return;
            }
        }
        result.dedupUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - dedupStartedAt).count());
    }

    nnue_eval_shared::EvaluatedPosition evaluated;
    std::string evalError;
    if (!nnue_eval_shared::evaluate_position_with_optional_pv(
          make_eval_options(opt),
          pos,
          networks,
          caches,
          accumulators,
          nullptr,
          nullptr,
          evaluated,
          evalError))
    {
        result.fatalError = evalError;
        return;
    }

    const auto rowBuildStartedAt = std::chrono::steady_clock::now();
    RowData row;
    row.fen = std::move(fen);
    row.pieceCount = pieceCount;
    row.psqt = evaluated.root.psqt;
    row.positional = evaluated.root.positional;
    row.nnue = evaluated.root.nnue;
    row.minElo = game.minElo;
    row.source = game.source;
    result.rowsByBucket[static_cast<std::size_t>(bucket)].push_back(std::move(row));
    result.rowBuildUs += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - rowBuildStartedAt).count());
    result.positionsSelected++;
}

bool read_games_chunk(std::istream& in,
                      std::vector<GameInput>& games,
                      std::size_t             maxGames,
                      std::size_t&            lineNo,
                      std::size_t&            selectedFull,
                      std::size_t&            selectedEndgameOnly,
                      std::string&            err) {
    games.clear();
    if (maxGames == 0)
        maxGames = 1;

    std::string line;
    while (games.size() < maxGames && std::getline(in, line))
    {
        ++lineNo;
        if (trim_ascii(line).empty())
            continue;

        GameInput game;
        std::string parseErr;
        if (!parse_record_line(line, game, parseErr))
        {
            err = "Line " + std::to_string(lineNo) + ": " + parseErr;
            return false;
        }
        if (game.includeFullGame)
            ++selectedFull;
        else
            ++selectedEndgameOnly;
        games.push_back(std::move(game));
    }

    return true;
}

inline int bucket_index(int pieceCount, Stockfish::Color stm) {
    const int stmBlack = stm == Stockfish::BLACK ? 1 : 0;
    const int pieceBucket8 = std::min(7, std::max(0, (std::max(pieceCount, 1) - 1) / 4));
    return pieceBucket8 * 2 + stmBlack;
}

std::uint64_t hash_fen64(std::string_view text, std::uint64_t seed) {
    std::uint64_t h = seed;
    for (unsigned char c : text)
    {
        h ^= c;
        h *= FnvPrime64;
    }
    return h;
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
    return !key.sideToMove.empty();
}

std::uint64_t hash_nnue_key64(const NnueFenKeyView& key, std::uint64_t seed) {
    std::uint64_t h = hash_fen64(key.board, seed);
    h ^= ' ';
    h *= FnvPrime64;
    h = hash_fen64(key.sideToMove, h);
    return h;
}


std::string json_escape_string(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (unsigned char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20)
            {
                static constexpr char Hex[] = "0123456789abcdef";
                out += "\\u00";
                out += Hex[(c >> 4) & 0x0F];
                out += Hex[c & 0x0F];
            }
            else
                out += static_cast<char>(c);
        }
    }
    return out;
}


bool parse_bucket_key(const std::string& text, std::size_t& bucket) {
    if (text.size() != 3 || text[0] != 'b')
        return false;
    if (!std::isdigit(static_cast<unsigned char>(text[1]))
        || !std::isdigit(static_cast<unsigned char>(text[2])))
        return false;
    bucket = static_cast<std::size_t>((text[1] - '0') * 10 + (text[2] - '0'));
    return bucket < BucketCount;
}

bool parse_uint64_text(const std::string& text, std::uint64_t& value) {
    if (text.empty())
        return false;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool extract_csv_fen_field(const std::string& line, std::string& fen) {
    if (line.empty())
        return false;

    if (line.front() == '"')
    {
        const std::size_t quoteEnd = line.find('"', 1);
        if (quoteEnd == std::string::npos)
            return false;
        fen = line.substr(1, quoteEnd - 1);
        return !fen.empty();
    }

    const std::size_t comma = line.find(',');
    fen = comma == std::string::npos ? line : line.substr(0, comma);
    fen = trim_ascii(fen);
    return !fen.empty();
}

bool load_hashes_from_csv(const std::filesystem::path& path,
                          FlatHashSet64& seen,
                          std::size_t& count,
                          std::array<std::size_t, 33>& pieceCountCounts,
                          std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "Failed to open existing CSV for append preload: " + path.string();
        return false;
    }

    std::string line;
    bool firstLine = true;
    while (std::getline(in, line))
    {
        if (firstLine)
        {
            firstLine = false;
            continue;
        }
        if (trim_ascii(line).empty())
            continue;

        std::string fen;
        if (!extract_csv_fen_field(line, fen))
        {
            err = "Failed to parse existing CSV FEN field: " + path.string();
            return false;
        }

        NnueFenKeyView key;
        if (!extract_nnue_fen_key(fen, key))
        {
            err = "Failed to parse existing CSV FEN for dedup preload: " + path.string();
            return false;
        }

        seen.insert(hash_nnue_key64(key, FnvOffsetBasis64A));
        count++;

        const auto pieceCount = static_cast<std::size_t>(std::count_if(
          key.board.begin(), key.board.end(), [](char c) { return std::isalpha(static_cast<unsigned char>(c)); }));
        if (pieceCount < pieceCountCounts.size())
            pieceCountCounts[pieceCount]++;
    }

    return true;
}

bool load_exclude_hash_files(const std::vector<std::string>& files,
                             std::array<FlatHashSet64, BucketCount>& seen64,
                             std::size_t& loaded,
                             std::string& err) {
    for (const auto& file : files)
    {
        std::ifstream in(file, std::ios::binary);
        if (!in)
        {
            err = "Failed to open exclude hash file: " + file;
            return false;
        }

        std::string line;
        std::size_t lineNo = 0;
        while (std::getline(in, line))
        {
            ++lineNo;
            auto work = trim_ascii(line);
            if (work.empty() || work[0] == '#')
                continue;

            const auto comma = work.find(',');
            if (comma == std::string::npos)
            {
                err = "Invalid exclude hash record at " + file + ":" + std::to_string(lineNo);
                return false;
            }

            const std::string bucketText = trim_ascii(work.substr(0, comma));
            const std::string hashText = trim_ascii(work.substr(comma + 1));
            std::size_t bucket = 0;
            std::uint64_t hash = 0;
            if (!parse_bucket_key(bucketText, bucket) || !parse_uint64_text(hashText, hash))
            {
                err = "Invalid exclude hash record at " + file + ":" + std::to_string(lineNo);
                return false;
            }

            seen64[bucket].insert(hash);
            loaded++;
        }
    }
    return true;
}

nnue_eval_shared::EvalOptions make_eval_options(const Options& opt) {
    nnue_eval_shared::EvalOptions shared;
    shared.outputNet = nnue_eval_shared::OutputNet::Big;
    shared.unscaledOutput = true;
    shared.nnueLabelMode = opt.nnueLabelMode;
    return shared;
}

GameProcessResult process_game(const GameInput& game,
                               const Options&   opt,
                               Stockfish::Eval::NNUE::Networks&          networks,
                               Stockfish::Eval::NNUE::AccumulatorCaches& caches,
                               Stockfish::Eval::NNUE::AccumulatorStack&  accumulators,
                               SharedDedupState*                         dedupState) {
    using namespace Stockfish;

    GameProcessResult result;
    result.phase = game.phase.empty() ? opt.phaseLabel : game.phase;
    const auto setupStartedAt = std::chrono::steady_clock::now();
    Position pos;
    std::vector<StateInfo> states(game.moves.size() + 1);
    pos.set(game.startFen.empty() ? StartFEN : game.startFen, false, &states[0]);
    accumulators.reset();
    result.gameSetupUs += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - setupStartedAt).count());
    int accumPly = 0;
    int currentPly = 0;

    append_current_position(
      game, opt, pos, networks, caches, accumulators, dedupState, currentPly, result);
    if (!result.fatalError.empty())
        return result;

    if (game.moves.empty())
    {
        result.emptyGame = game.startFen.empty();
        return result;
    }

    for (std::size_t i = 0; i < game.moves.size(); ++i)
    {
        const auto parseStartedAt = std::chrono::steady_clock::now();
        Move m = UCIEngine::to_move(pos, game.moves[i]);
        result.moveParseUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - parseStartedAt).count());
        if (m == Move::none())
        {
            result.invalidGame = true;
            return result;
        }

        const auto applyStartedAt = std::chrono::steady_clock::now();
        if (accumPly + 1 >= int(Stockfish::Eval::NNUE::AccumulatorStack::MaxSize))
        {
            accumulators.reset();
            accumPly = 0;
        }

        auto [dirtyPiece, dirtyThreats] = accumulators.push();
        pos.do_move(m, states[i + 1], pos.gives_check(m), dirtyPiece, dirtyThreats, nullptr,
                    nullptr);
        accumPly++;
        currentPly++;
        result.moveApplyUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - applyStartedAt).count());
        append_current_position(
          game, opt, pos, networks, caches, accumulators, dedupState, currentPly, result);
        if (!result.fatalError.empty())
            return result;
    }

    return result;
}

BatchResult process_games_chunk(const std::vector<GameInput>& games,
                                 const Options&               opt,
                                 Stockfish::Eval::NNUE::Networks& networks,
                                 WorkerResources&                workerResources,
                                 SharedDedupState*               dedupState) {
    BatchResult result;
    if (games.empty())
        return result;

    const auto batchStartedAt = std::chrono::steady_clock::now();
    if (!workerResources.initialized)
    {
        workerResources.caches = std::make_unique<Stockfish::Eval::NNUE::AccumulatorCaches>(networks);
        workerResources.accumulators = std::make_unique<Stockfish::Eval::NNUE::AccumulatorStack>();
        workerResources.initialized = true;
    }
    const auto setupFinishedAt = std::chrono::steady_clock::now();
    result.stats.batchSetupUs = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(setupFinishedAt - batchStartedAt).count());

    for (const GameInput& g : games)
    {
        const auto gr =
          process_game(g,
                       opt,
                        networks,
                        *workerResources.caches,
                        *workerResources.accumulators,
                        dedupState);
        if (!gr.fatalError.empty())
        {
            result.fatalError = gr.fatalError;
            return result;
        }
        result.stats.gamesProcessed++;
        if (g.includeFullGame)
            result.stats.gamesFull++;
        else
            result.stats.gamesEndgameOnly++;

        if (gr.emptyGame)
            result.stats.gamesSkippedNoMoves++;
        if (gr.invalidGame)
            result.stats.gamesSkippedInvalid++;

        result.stats.positionsScanned += gr.positionsScanned;
        result.stats.positionsSelected += gr.positionsSelected;
        result.stats.duplicateSkipped += gr.duplicateSkipped;
        auto& phaseStats = result.stats.phaseStats[gr.phase];
        phaseStats.positionsScanned += gr.positionsScanned;
        phaseStats.positionsWritten += gr.positionsSelected;
        phaseStats.duplicateSkipped += gr.duplicateSkipped;
        if (result.stats.phaseSegments.empty() || result.stats.phaseSegments.back().first != gr.phase)
            result.stats.phaseSegments.push_back({gr.phase, {}});
        PhaseStats& phaseSegmentStats = result.stats.phaseSegments.back().second;
        phaseSegmentStats.positionsScanned += gr.positionsScanned;
        phaseSegmentStats.positionsWritten += gr.positionsSelected;
        phaseSegmentStats.duplicateSkipped += gr.duplicateSkipped;
        result.stats.gameSetupUs += gr.gameSetupUs;
        result.stats.moveParseUs += gr.moveParseUs;
        result.stats.moveApplyUs += gr.moveApplyUs;
        result.stats.fenSerializeUs += gr.fenSerializeUs;
        result.stats.dedupUs += gr.dedupUs;
        result.stats.rowBuildUs += gr.rowBuildUs;

        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            auto& dst = result.rowsByBucket[bucket];
            const auto& src = gr.rowsByBucket[bucket];
            dst.insert(dst.end(), src.begin(), src.end());
        }
    }

    result.stats.batchProcessUs = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - setupFinishedAt).count());

    return result;
}

std::string bucket_key(std::size_t bucket) {
    std::ostringstream oss;
    oss << 'b' << std::setw(2) << std::setfill('0') << bucket;
    return oss.str();
}

bool write_count_cache(const std::filesystem::path& path,
                       std::size_t                  totalRows,
                       const std::array<std::size_t, 33>& pieceCountCounts,
                       std::string&                 err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "Failed to open count cache for writing: " + path.string();
        return false;
    }

    out << "{\n  \"total_rows\": " << totalRows << ",\n  \"piece_count_counts\": {\n";
    bool first = true;
    for (std::size_t pieceCount = 0; pieceCount < pieceCountCounts.size(); ++pieceCount)
    {
        if (pieceCountCounts[pieceCount] == 0)
            continue;
        if (!first)
            out << ",\n";
        out << "    \"" << pieceCount << "\": " << pieceCountCounts[pieceCount];
        first = false;
    }
    out << "\n  }\n}\n";
    return true;
}

bool read_count_cache(const std::filesystem::path& path,
                      std::size_t&                 totalRows,
                      std::array<std::size_t, 33>& pieceCountCounts,
                      std::string&                 err) {
    totalRows = 0;
    pieceCountCounts.fill(0);

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "Failed to open count cache for reading: " + path.string();
        return false;
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    const std::string totalKey = "\"total_rows\"";
    const auto totalPos = text.find(totalKey);
    if (totalPos == std::string::npos)
    {
        err = "Missing total_rows in count cache: " + path.string();
        return false;
    }
    const auto totalColon = text.find(':', totalPos + totalKey.size());
    if (totalColon == std::string::npos)
    {
        err = "Malformed total_rows in count cache: " + path.string();
        return false;
    }
    std::size_t totalStart = totalColon + 1;
    while (totalStart < text.size() && std::isspace(static_cast<unsigned char>(text[totalStart])))
        ++totalStart;
    std::size_t totalEnd = totalStart;
    while (totalEnd < text.size() && std::isdigit(static_cast<unsigned char>(text[totalEnd])))
        ++totalEnd;
    if (totalStart == totalEnd)
    {
        err = "Invalid total_rows in count cache: " + path.string();
        return false;
    }
    totalRows = static_cast<std::size_t>(std::stoull(text.substr(totalStart, totalEnd - totalStart)));

    const std::string pieceKey = "\"piece_count_counts\"";
    const auto piecePos = text.find(pieceKey);
    if (piecePos == std::string::npos)
        return true;
    const auto pieceOpen = text.find('{', piecePos + pieceKey.size());
    const auto pieceClose = text.find('}', pieceOpen == std::string::npos ? piecePos : pieceOpen + 1);
    if (pieceOpen == std::string::npos || pieceClose == std::string::npos)
    {
        err = "Malformed piece_count_counts in count cache: " + path.string();
        return false;
    }

    std::size_t pos = pieceOpen + 1;
    while (pos < pieceClose)
    {
        const auto keyOpen = text.find('"', pos);
        if (keyOpen == std::string::npos || keyOpen >= pieceClose)
            break;
        const auto keyClose = text.find('"', keyOpen + 1);
        if (keyClose == std::string::npos || keyClose >= pieceClose)
        {
            err = "Malformed piece_count_counts key in count cache: " + path.string();
            return false;
        }
        const auto colon = text.find(':', keyClose + 1);
        if (colon == std::string::npos || colon >= pieceClose)
        {
            err = "Malformed piece_count_counts value in count cache: " + path.string();
            return false;
        }
        std::size_t valueStart = colon + 1;
        while (valueStart < pieceClose && std::isspace(static_cast<unsigned char>(text[valueStart])))
            ++valueStart;
        std::size_t valueEnd = valueStart;
        while (valueEnd < pieceClose && std::isdigit(static_cast<unsigned char>(text[valueEnd])))
            ++valueEnd;
        int pieceCount = 0;
        int count = 0;
        if (!parse_nonnegative_int(text.substr(keyOpen + 1, keyClose - keyOpen - 1), pieceCount)
            || !parse_nonnegative_int(text.substr(valueStart, valueEnd - valueStart), count))
        {
            err = "Invalid piece_count_counts entry in count cache: " + path.string();
            return false;
        }
        if (pieceCount >= 0 && pieceCount < static_cast<int>(pieceCountCounts.size()))
            pieceCountCounts[static_cast<std::size_t>(pieceCount)] = static_cast<std::size_t>(count);
        pos = valueEnd;
    }

    return true;
}

bool flush_outputs(std::vector<std::ofstream>& outputs,
                   const std::vector<std::filesystem::path>& countPaths,
                   const std::array<std::size_t, BucketCount>& bucketCounts,
                   const std::array<std::array<std::size_t, 33>, BucketCount>& pieceCountCounts,
                   bool writeCountCaches,
                   std::string& err) {
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        outputs[bucket].flush();
        if (!outputs[bucket])
        {
            err = "Failed to flush output CSV";
            return false;
        }
        if (writeCountCaches && !write_count_cache(countPaths[bucket], bucketCounts[bucket], pieceCountCounts[bucket], err))
            return false;
    }
    return true;
}

double total_size_mb(const std::vector<std::filesystem::path>& csvPaths) {
    std::uintmax_t totalBytes = 0;
    std::error_code ec;
    for (const auto& path : csvPaths)
    {
        ec.clear();
        if (std::filesystem::is_regular_file(path, ec) && !ec)
        {
            ec.clear();
            const auto size = std::filesystem::file_size(path, ec);
            if (!ec)
                totalBytes += size;
        }
    }
    return static_cast<double>(totalBytes) / (1024.0 * 1024.0);
}

std::uint64_t hash_row64(const RowData& row) {
    std::uint64_t h = FnvOffsetBasis64A;
    h = hash_fen64(row.fen, h);
    h ^= '\t';
    h *= FnvPrime64;
    const std::string psqt = std::to_string(int(row.psqt));
    h = hash_fen64(psqt, h);
    h ^= '\t';
    h *= FnvPrime64;
    const std::string positional = std::to_string(int(row.positional));
    h = hash_fen64(positional, h);
    h ^= '\t';
    h *= FnvPrime64;
    const std::string nnue = std::to_string(int(row.nnue));
    h = hash_fen64(nnue, h);
    return h;
}

std::filesystem::path split_shard_root(const std::filesystem::path& outDir, const std::string& outputStem) {
    return outDir / (".split_tmp_" + outputStem);
}

std::filesystem::path split_shard_path(const std::filesystem::path& root, std::size_t bucket, std::size_t shard) {
    std::ostringstream name;
    name << "shard_" << std::setw(4) << std::setfill('0') << shard << ".tsv";
    return root / bucket_key(bucket) / name.str();
}

std::filesystem::path split_csv_path(const std::filesystem::path& outDir,
                                     const std::string& outputStem,
                                     const std::string& splitName,
                                     std::size_t        bucket) {
    return outDir / (outputStem + "_" + splitName + "_" + bucket_key(bucket) + ".csv");
}

bool append_text_file(const std::filesystem::path& path, const std::string& data, std::string& err) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
    {
        err = "Failed to create shard directory: " + path.parent_path().string();
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out)
    {
        err = "Failed to open shard file for append: " + path.string();
        return false;
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out)
    {
        err = "Failed writing shard file: " + path.string();
        return false;
    }
    return true;
}

bool flush_shard_buffers(
  const std::filesystem::path&                                shardRoot,
  const std::array<std::vector<std::string>, BucketCount>&    shardBuffers,
  std::string&                                                err) {
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        for (std::size_t shard = 0; shard < shardBuffers[bucket].size(); ++shard)
        {
            if (shardBuffers[bucket][shard].empty())
                continue;
            if (!append_text_file(split_shard_path(shardRoot, bucket, shard), shardBuffers[bucket][shard], err))
                return false;
        }
    }
    return true;
}

void clear_shard_buffers(std::array<std::vector<std::string>, BucketCount>& shardBuffers) {
    for (auto& perBucket : shardBuffers)
    {
        for (auto& buffer : perBucket)
            buffer.clear();
    }
}

struct AllocationRemainder {
    double      remainder = 0.0;
    std::size_t spare     = 0;
    std::size_t bucket    = 0;
};

std::array<std::size_t, BucketCount> allocate_proportional(
  std::size_t                                       totalTarget,
  const std::array<std::size_t, BucketCount>&       capacities,
  const std::array<std::size_t, BucketCount>&       weights) {
    std::array<std::size_t, BucketCount> allocation{};
    const std::size_t totalCapacity =
      std::accumulate(capacities.begin(), capacities.end(), std::size_t{0});
    if (totalTarget == 0 || totalCapacity == 0)
        return allocation;

    const std::size_t target = std::min(totalTarget, totalCapacity);
    std::size_t totalWeight = 0;
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        totalWeight += weights[bucket];
    if (totalWeight == 0)
        return allocation;

    std::vector<AllocationRemainder> remainders;
    std::size_t allocated = 0;
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        const double exact = static_cast<double>(target) * static_cast<double>(weights[bucket])
                           / static_cast<double>(totalWeight);
        const std::size_t base =
          std::min(capacities[bucket], static_cast<std::size_t>(std::floor(exact)));
        allocation[bucket] = base;
        allocated += base;
        if (base < capacities[bucket])
        {
            remainders.push_back(
              {exact - std::floor(exact), capacities[bucket] - base, bucket});
        }
    }

    std::size_t remaining = target - allocated;
    std::sort(remainders.begin(), remainders.end(), [](const AllocationRemainder& a, const AllocationRemainder& b) {
        if (a.remainder != b.remainder)
            return a.remainder > b.remainder;
        if (a.spare != b.spare)
            return a.spare > b.spare;
        return a.bucket < b.bucket;
    });

    while (remaining > 0)
    {
        bool progress = false;
        for (const auto& item : remainders)
        {
            if (allocation[item.bucket] >= capacities[item.bucket])
                continue;
            allocation[item.bucket]++;
            remaining--;
            progress = true;
            if (remaining == 0)
                break;
        }
        if (!progress)
            break;
    }

    return allocation;
}

struct ShardRow {
    std::uint64_t hash = 0;
    int           pieceCount = 0;
    std::string   fen;
    int           psqt = 0;
    int           positional = 0;
    int           nnue = 0;
    std::optional<int> minElo;
    std::string   source;
};

bool parse_uint64_hex(const std::string& text, std::uint64_t& value) {
    if (text.empty())
        return false;
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(text.c_str(), &end, 16);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;
    value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parse_int_text(const std::string& text, int& value) {
    if (text.empty())
        return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0')
        return false;
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
        return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_optional_min_elo_field(const std::string& text, std::optional<int>& minElo) {
    minElo.reset();
    if (trim_ascii(text).empty())
        return true;
    int parsed = 0;
    if (!parse_nonnegative_int(trim_ascii(text), parsed))
        return false;
    minElo = parsed;
    return true;
}

bool parse_shard_line(const std::string& line, ShardRow& row) {
    std::vector<std::string_view> fields;
    fields.reserve(13);
    std::size_t start = 0;
    while (true)
    {
        const std::size_t tab = line.find('\t', start);
        if (tab == std::string::npos)
        {
            fields.push_back(std::string_view(line).substr(start));
            break;
        }
        fields.push_back(std::string_view(line).substr(start, tab - start));
        start = tab + 1;
    }
    if (fields.size() != 8)
        return false;

    if (!parse_uint64_hex(std::string(fields[0]), row.hash))
        return false;
    if (!parse_int_text(std::string(fields[1]), row.pieceCount))
        return false;
    row.fen.assign(fields[2].data(), fields[2].size());
    if (!parse_int_text(std::string(fields[3]), row.psqt))
        return false;
    if (!parse_int_text(std::string(fields[4]), row.positional))
        return false;
    if (!parse_int_text(std::string(fields[5]), row.nnue))
        return false;
    if (!parse_optional_min_elo_field(std::string(fields[6]), row.minElo))
        return false;
    row.source.assign(fields[7].data(), fields[7].size());
    return true;
}

bool parse_bucket_csv_line(const std::string& line, ShardRow& row) {
    if (trim_ascii(line).empty())
        return false;

    std::string fen;
    std::size_t cursor = 0;
    if (!line.empty() && line.front() == '"')
    {
        const auto quoteEnd = line.find('"', 1);
        if (quoteEnd == std::string::npos)
            return false;
        fen = line.substr(1, quoteEnd - 1);
        cursor = quoteEnd + 1;
        if (cursor >= line.size() || line[cursor] != ',')
            return false;
        ++cursor;
    }
    else
    {
        const auto comma = line.find(',');
        if (comma == std::string::npos)
            return false;
        fen = trim_ascii(line.substr(0, comma));
        cursor = comma + 1;
    }

    row.fen = std::move(fen);
    std::vector<std::string> fields;
    while (cursor <= line.size())
    {
        const auto comma = line.find(',', cursor);
        if (comma == std::string::npos)
        {
            fields.push_back(trim_ascii(line.substr(cursor)));
            break;
        }
        fields.push_back(trim_ascii(line.substr(cursor, comma - cursor)));
        cursor = comma + 1;
    }
    if (fields.size() != 5)
        return false;
    if (!parse_int_text(fields[0], row.psqt))
        return false;
    if (!parse_int_text(fields[1], row.positional))
        return false;
    if (!parse_int_text(fields[2], row.nnue))
        return false;
    if (!parse_optional_min_elo_field(fields[3], row.minElo))
        return false;
    row.source = fields[4];
    row.pieceCount = static_cast<int>(std::count_if(
      row.fen.begin(), row.fen.end(), [](char c) { return std::isalpha(static_cast<unsigned char>(c)); }));
    row.hash = 0;
    return true;
}

struct SplitFinalizeResult {
    std::array<std::array<std::filesystem::path, BucketCount>, 3> paths;
    std::array<std::array<std::size_t, BucketCount>, 3>           counts{};
    std::array<std::array<std::array<std::size_t, 33>, BucketCount>, 3> pieceCounts{};
    std::array<std::size_t, 3> totals{};
    double elapsedSec = 0.0;
    double finalSizeMb = 0.0;
};

struct SplitRuntimeProgress {
    std::atomic<std::size_t> rowsLoaded{0};
    std::atomic<std::size_t> rowsWritten{0};
    std::atomic<std::size_t> trainWritten{0};
    std::atomic<std::size_t> validationWritten{0};
    std::atomic<std::size_t> testWritten{0};
    std::atomic<std::size_t> bucketsCompleted{0};
    std::atomic<std::size_t> activeBuckets{0};
};

bool load_existing_split_bucket_outputs(const Options&                              opt,
                                        const std::filesystem::path&                outDir,
                                        std::size_t                                 bucket,
                                        std::size_t                                 bucketTotalRows,
                                        std::array<std::size_t, 3>&                 splitCounts,
                                        std::array<std::array<std::size_t, 33>, 3>& splitPieceCounts,
                                        std::string&                                err) {
    static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
    std::size_t totalRows = 0;
    for (std::size_t split = 0; split < splitNames.size(); ++split)
    {
        const auto csvPath = split_csv_path(outDir, opt.outputStem, splitNames[split], bucket);
        const auto countPath = std::filesystem::path(csvPath.string() + ".count");
        if (!std::filesystem::exists(csvPath) || !std::filesystem::exists(countPath))
            return false;
        std::size_t count = 0;
        if (!read_count_cache(countPath, count, splitPieceCounts[split], err))
            return false;
        splitCounts[split] = count;
        totalRows += count;
    }
    if (totalRows != bucketTotalRows)
        return false;
    return true;
}

bool split_single_bucket_in_memory(const Options&                              opt,
                                   const std::filesystem::path&                outDir,
                                   const std::filesystem::path&                csvPath,
                                   std::size_t                                 bucket,
                                   std::size_t                                 bucketTotalRows,
                                   std::size_t                                 /*trainQuota*/,
                                   std::size_t                                 validationQuota,
                                   std::size_t                                 testQuota,
                                   std::array<std::size_t, 3>&                 splitCounts,
                                   std::array<std::array<std::size_t, 33>, 3>& splitPieceCounts,
                                   SplitRuntimeProgress*                       progress,
                                   std::string&                                err) {
    std::ifstream in(csvPath, std::ios::binary);
    if (!in)
    {
        err = "Failed to open bucket CSV for split: " + csvPath.string();
        return false;
    }

    std::vector<ShardRow> rows;
    rows.reserve(bucketTotalRows);
    std::string line;
    bool firstLine = true;
    while (std::getline(in, line))
    {
        if (firstLine)
        {
            firstLine = false;
            continue;
        }
        if (trim_ascii(line).empty())
            continue;
        ShardRow row;
        if (!parse_bucket_csv_line(line, row))
        {
            err = "Failed to parse bucket CSV line: " + csvPath.string();
            return false;
        }
        rows.push_back(std::move(row));
        if (progress)
            progress->rowsLoaded.fetch_add(1, std::memory_order_relaxed);
    }

    if (rows.size() != bucketTotalRows)
    {
        err = "Bucket CSV row count mismatch for " + bucket_key(bucket);
        return false;
    }

    std::vector<std::uint32_t> indices(rows.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::random_device rd;
    const auto nowSeed = static_cast<std::uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng((static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd()) ^ nowSeed
                        ^ static_cast<std::uint64_t>(bucket));
    std::shuffle(indices.begin(), indices.end(), rng);

    static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
    std::array<std::ofstream, 3> outputs;
    for (std::size_t split = 0; split < splitNames.size(); ++split)
    {
        const auto outPath = split_csv_path(outDir, opt.outputStem, splitNames[split], bucket);
        std::error_code ec;
        std::filesystem::remove(outPath, ec);
        std::filesystem::remove(std::filesystem::path(outPath.string() + ".count"), ec);
        outputs[split].open(outPath, std::ios::binary | std::ios::trunc);
        if (!outputs[split])
        {
            err = "Failed to open split output CSV: " + outPath.string();
            return false;
        }
        outputs[split] << output_csv_header(opt);
    }

    std::size_t processedRows = 0;
    std::size_t remainingValidation = validationQuota;
    std::size_t remainingTest = testQuota;
    for (const std::uint32_t index : indices)
    {
        const auto& row = rows[index];
        std::size_t split = 0;
        if (remainingValidation > 0)
        {
            split = 1;
            --remainingValidation;
        }
        else if (remainingTest > 0)
        {
            split = 2;
            --remainingTest;
        }

        outputs[split] << '"' << row.fen << '"' << ','
                       << row.psqt << ','
                       << row.positional << ','
                       << row.nnue;
        outputs[split] << ',';
        if (row.minElo)
            outputs[split] << *row.minElo;
        outputs[split] << ',' << row.source;
        outputs[split] << '\n';
        if (!outputs[split])
        {
            err = "Failed writing split output CSV for " + bucket_key(bucket);
            return false;
        }
        splitCounts[split]++;
        if (row.pieceCount >= 0 && row.pieceCount < static_cast<int>(splitPieceCounts[split].size()))
            splitPieceCounts[split][static_cast<std::size_t>(row.pieceCount)]++;
        ++processedRows;
        if (progress)
        {
            progress->rowsWritten.fetch_add(1, std::memory_order_relaxed);
            if (split == 0)
                progress->trainWritten.fetch_add(1, std::memory_order_relaxed);
            else if (split == 1)
                progress->validationWritten.fetch_add(1, std::memory_order_relaxed);
            else
                progress->testWritten.fetch_add(1, std::memory_order_relaxed);
        }
    }

    for (std::size_t split = 0; split < splitNames.size(); ++split)
    {
        outputs[split].flush();
        outputs[split].close();
        const auto outPath = split_csv_path(outDir, opt.outputStem, splitNames[split], bucket);
        if (!write_count_cache(std::filesystem::path(outPath.string() + ".count"),
                               splitCounts[split],
                               splitPieceCounts[split],
                               err))
            return false;
    }

    if (remainingValidation != 0 || remainingTest != 0)
    {
        err = "Split quotas not satisfied for " + bucket_key(bucket);
        return false;
    }

    return true;
}

bool finalize_existing_bucket_csvs(const Options&                                   opt,
                                   const std::filesystem::path&                     outDir,
                                   const std::array<std::size_t, BucketCount>&      bucketCounts,
                                   SplitFinalizeResult&                             result,
                                   std::string&                                     err) {
    static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
    const auto startedAt = std::chrono::steady_clock::now();
    const std::size_t totalRows = std::accumulate(bucketCounts.begin(), bucketCounts.end(), std::size_t{0});
    const std::size_t validationTarget =
      std::min(opt.maxValidationRows,
               static_cast<std::size_t>(std::floor(static_cast<double>(totalRows) * opt.validationRatio)));
    const std::size_t testTarget =
      std::min(opt.maxTestRows,
               static_cast<std::size_t>(std::floor(static_cast<double>(totalRows) * opt.testRatio)));
    if (validationTarget + testTarget > totalRows)
    {
        err = "Requested validation+test rows exceed total rows";
        return false;
    }

    const auto validationQuota = allocate_proportional(validationTarget, bucketCounts, bucketCounts);
    std::array<std::size_t, BucketCount> remainingAfterValidation{};
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        remainingAfterValidation[bucket] = bucketCounts[bucket] - validationQuota[bucket];
    const auto testQuota = allocate_proportional(testTarget, remainingAfterValidation, bucketCounts);
    std::array<std::size_t, BucketCount> trainQuota{};
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        trainQuota[bucket] = bucketCounts[bucket] - validationQuota[bucket] - testQuota[bucket];

    std::vector<std::filesystem::path> finalPaths;
    finalPaths.reserve(BucketCount * splitNames.size());
    for (std::size_t split = 0; split < splitNames.size(); ++split)
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            result.paths[split][bucket] = split_csv_path(outDir, opt.outputStem, splitNames[split], bucket);
            finalPaths.push_back(result.paths[split][bucket]);
        }

    std::atomic<std::size_t> nextBucket{0};
    std::atomic<std::size_t> completedRows{0};
    SplitRuntimeProgress runtimeProgress;
    std::mutex resultMutex;
    std::mutex errorMutex;
    std::string firstError;
    std::atomic<bool> failed{false};
    std::atomic<bool> monitorStop{false};

    std::thread monitorThread([&]() {
        const auto monitorStartedAt = std::chrono::steady_clock::now();
        auto nextReportAt = monitorStartedAt + std::chrono::seconds(30);
        while (!monitorStop.load())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (monitorStop.load())
                break;
            const auto now = std::chrono::steady_clock::now();
            if (now < nextReportAt)
                continue;
            nextReportAt += std::chrono::seconds(30);
            const auto written = runtimeProgress.rowsWritten.load(std::memory_order_relaxed);
            const auto loaded = runtimeProgress.rowsLoaded.load(std::memory_order_relaxed);
            const auto trainWritten = runtimeProgress.trainWritten.load(std::memory_order_relaxed);
            const auto validationWritten = runtimeProgress.validationWritten.load(std::memory_order_relaxed);
            const auto testWritten = runtimeProgress.testWritten.load(std::memory_order_relaxed);
            const auto completedBuckets = runtimeProgress.bucketsCompleted.load(std::memory_order_relaxed);
            const auto activeBuckets = runtimeProgress.activeBuckets.load(std::memory_order_relaxed);
            const double elapsed =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - monitorStartedAt).count();
            const double rate = elapsed > 0.0 ? static_cast<double>(written) / elapsed : 0.0;
            std::cout << "[split-summary] read " << format_grouped_u64(loaded)
                      << "/" << format_grouped_u64(totalRows)
                      << " | total " << format_grouped_u64(written)
                      << "/" << format_grouped_u64(totalRows)
                      << " | train " << format_grouped_u64(trainWritten)
                      << " | validation " << format_grouped_u64(validationWritten)
                      << " | test " << format_grouped_u64(testWritten)
                      << " | completed_buckets " << format_grouped_u64(completedBuckets)
                      << "/" << BucketCount
                      << " | active_buckets " << format_grouped_u64(activeBuckets)
                      << " | rate " << format_grouped_u64(static_cast<std::uint64_t>(std::llround(rate))) << " positions/s\n";
            std::cout.flush();
        }
    });

    auto worker = [&]() {
        while (true)
        {
            const std::size_t bucket = nextBucket.fetch_add(1);
            if (bucket >= BucketCount || failed.load())
                break;
            runtimeProgress.activeBuckets.fetch_add(1, std::memory_order_relaxed);

            std::array<std::size_t, 3> splitCounts{};
            std::array<std::array<std::size_t, 33>, 3> splitPieceCounts{};

            std::string localErr;
            if (!load_existing_split_bucket_outputs(opt, outDir, bucket, bucketCounts[bucket], splitCounts, splitPieceCounts, localErr))
            {
                const auto csvPath = outDir / (opt.outputStem + "_" + bucket_key(bucket) + ".csv");
                if (!std::filesystem::exists(csvPath))
                {
                    runtimeProgress.activeBuckets.fetch_sub(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (!failed.exchange(true))
                        firstError = "Temporary bucket CSV not found for " + bucket_key(bucket) + ": " + csvPath.string();
                    break;
                }
                if (!split_single_bucket_in_memory(opt,
                                                   outDir,
                                                   csvPath,
                                                   bucket,
                                                   bucketCounts[bucket],
                                                   trainQuota[bucket],
                                                   validationQuota[bucket],
                                                   testQuota[bucket],
                                                   splitCounts,
                                                   splitPieceCounts,
                                                   &runtimeProgress,
                                                   localErr))
                {
                    runtimeProgress.activeBuckets.fetch_sub(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (!failed.exchange(true))
                        firstError = std::move(localErr);
                    break;
                }
                std::error_code ec;
                std::filesystem::remove(csvPath, ec);
                std::filesystem::remove(std::filesystem::path(csvPath.string() + ".count"), ec);
            }
            else
            {
                runtimeProgress.rowsWritten.fetch_add(bucketCounts[bucket], std::memory_order_relaxed);
                runtimeProgress.trainWritten.fetch_add(splitCounts[0], std::memory_order_relaxed);
                runtimeProgress.validationWritten.fetch_add(splitCounts[1], std::memory_order_relaxed);
                runtimeProgress.testWritten.fetch_add(splitCounts[2], std::memory_order_relaxed);
            }

            {
                std::lock_guard<std::mutex> lock(resultMutex);
                for (std::size_t split = 0; split < splitNames.size(); ++split)
                {
                    result.counts[split][bucket] = splitCounts[split];
                    result.pieceCounts[split][bucket] = splitPieceCounts[split];
                    result.totals[split] += splitCounts[split];
                }
            }
            runtimeProgress.activeBuckets.fetch_sub(1, std::memory_order_relaxed);
            runtimeProgress.bucketsCompleted.fetch_add(1, std::memory_order_relaxed);
            const auto completed = completedRows.fetch_add(bucketCounts[bucket]) + bucketCounts[bucket];
            const double elapsed =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
            const double rate = elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
            std::cout << "[split] " << bucket_key(bucket)
                      << " | total " << format_grouped_u64(bucketCounts[bucket])
                      << " | train " << format_grouped_u64(trainQuota[bucket])
                      << " | validation " << format_grouped_u64(validationQuota[bucket])
                      << " | test " << format_grouped_u64(testQuota[bucket])
                      << " | completed_rows " << format_grouped_u64(completed)
                      << " | rate " << format_grouped_u64(static_cast<std::uint64_t>(std::llround(rate))) << " positions/s\n";
            std::cout.flush();
        }
    };

    const std::size_t workerCount = std::max<std::size_t>(1, opt.splitWorkers);
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t i = 0; i < workerCount; ++i)
        workers.emplace_back(worker);
    for (auto& thread : workers)
        thread.join();
    monitorStop.store(true);
    monitorThread.join();

    if (failed.load())
    {
        err = firstError.empty() ? "Split finalization failed" : firstError;
        return false;
    }

    result.elapsedSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    result.finalSizeMb = total_size_mb(finalPaths);
    return true;
}

bool finalize_split_outputs(const Options&                                   opt,
                            const std::filesystem::path&                     outDir,
                            const std::filesystem::path&                     shardRoot,
                            const std::array<std::size_t, BucketCount>&      bucketCounts,
                            SplitFinalizeResult&                             result,
                            std::string&                                     err) {
    static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
    const auto startedAt = std::chrono::steady_clock::now();
    const std::size_t totalRows = std::accumulate(bucketCounts.begin(), bucketCounts.end(), std::size_t{0});
    const std::size_t validationTarget =
      std::min(opt.maxValidationRows,
               static_cast<std::size_t>(std::floor(static_cast<double>(totalRows) * opt.validationRatio)));
    const std::size_t testTarget =
      std::min(opt.maxTestRows,
               static_cast<std::size_t>(std::floor(static_cast<double>(totalRows) * opt.testRatio)));
    if (validationTarget + testTarget > totalRows)
    {
        err = "Requested validation+test rows exceed total rows";
        return false;
    }

    const auto validationQuota = allocate_proportional(validationTarget, bucketCounts, bucketCounts);
    std::array<std::size_t, BucketCount> remainingAfterValidation{};
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        remainingAfterValidation[bucket] = bucketCounts[bucket] - validationQuota[bucket];
    const auto testQuota = allocate_proportional(testTarget, remainingAfterValidation, bucketCounts);
    std::array<std::size_t, BucketCount> trainQuota{};
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        trainQuota[bucket] = bucketCounts[bucket] - validationQuota[bucket] - testQuota[bucket];

    std::array<std::array<std::ofstream, BucketCount>, 3> outputs;
    std::vector<std::filesystem::path> finalPaths;
    finalPaths.reserve(BucketCount * splitNames.size());

    for (std::size_t split = 0; split < splitNames.size(); ++split)
    {
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            result.paths[split][bucket] = split_csv_path(outDir, opt.outputStem, splitNames[split], bucket);
            finalPaths.push_back(result.paths[split][bucket]);
            std::error_code ec;
            std::filesystem::remove(result.paths[split][bucket], ec);
            std::filesystem::remove(std::filesystem::path(result.paths[split][bucket].string() + ".count"), ec);
            outputs[split][bucket].open(result.paths[split][bucket], std::ios::binary | std::ios::trunc);
            if (!outputs[split][bucket])
            {
                err = "Failed to open split output CSV: " + result.paths[split][bucket].string();
                return false;
            }
            outputs[split][bucket] << output_csv_header(opt);
        }
    }

    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        std::size_t remainingValidation = validationQuota[bucket];
        std::size_t remainingTest = testQuota[bucket];

        for (std::size_t shard = 0; shard < opt.splitShards; ++shard)
        {
            const auto path = split_shard_path(shardRoot, bucket, shard);
            std::error_code ec;
            if (!std::filesystem::exists(path, ec) || ec)
                continue;

            std::vector<ShardRow> rows;
            std::ifstream in(path, std::ios::binary);
            if (!in)
            {
                err = "Failed to open shard file: " + path.string();
                return false;
            }

            std::string line;
            while (std::getline(in, line))
            {
                if (line.empty())
                    continue;
                ShardRow row;
                if (!parse_shard_line(line, row))
                {
                    err = "Failed to parse shard line: " + path.string();
                    return false;
                }
                rows.push_back(std::move(row));
            }

            std::sort(rows.begin(), rows.end(), [](const ShardRow& a, const ShardRow& b) {
                return a.hash < b.hash;
            });

            for (const auto& row : rows)
            {
                std::size_t split = 0;
                if (remainingValidation > 0)
                {
                    split = 1;
                    --remainingValidation;
                }
                else if (remainingTest > 0)
                {
                    split = 2;
                    --remainingTest;
                }

                outputs[split][bucket] << '"' << row.fen << '"' << ','
                                       << row.psqt << ','
                                       << row.positional << ','
                                       << row.nnue;
                outputs[split][bucket] << ',';
                if (row.minElo)
                    outputs[split][bucket] << *row.minElo;
                outputs[split][bucket] << ',' << row.source;
                outputs[split][bucket] << '\n';
                if (!outputs[split][bucket])
                {
                    err = "Failed writing split output CSV: " + result.paths[split][bucket].string();
                    return false;
                }
                result.counts[split][bucket]++;
                result.totals[split]++;
                if (row.pieceCount >= 0 && row.pieceCount < static_cast<int>(result.pieceCounts[split][bucket].size()))
                    result.pieceCounts[split][bucket][static_cast<std::size_t>(row.pieceCount)]++;
            }
        }

        if (remainingValidation != 0 || remainingTest != 0)
        {
            err = "Split quotas not satisfied for " + bucket_key(bucket);
            return false;
        }

        std::cout << "[split] " << bucket_key(bucket)
                  << " | total " << format_grouped_u64(bucketCounts[bucket])
                  << " | train " << format_grouped_u64(trainQuota[bucket])
                  << " | validation " << format_grouped_u64(validationQuota[bucket])
                  << " | test " << format_grouped_u64(testQuota[bucket]) << "\n";
    }

    for (std::size_t split = 0; split < splitNames.size(); ++split)
    {
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            outputs[split][bucket].flush();
            outputs[split][bucket].close();
            if (!write_count_cache(std::filesystem::path(result.paths[split][bucket].string() + ".count"),
                                   result.counts[split][bucket],
                                   result.pieceCounts[split][bucket],
                                   err))
                return false;
        }
    }

    std::error_code cleanupEc;
    std::filesystem::remove_all(shardRoot, cleanupEc);
    result.elapsedSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    result.finalSizeMb = total_size_mb(finalPaths);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt))
    {
        print_usage();
        return 2;
    }
    if (opt.finalizeExistingSplit)
        opt.splitFinal = true;
    if (opt.splitFinal)
    {
        const double ratioSum = opt.trainRatio + opt.validationRatio + opt.testRatio;
        if (std::fabs(ratioSum - 1.0) > 1e-9)
        {
            std::cerr << "Split ratios must sum to 1.0, got " << ratioSum << "\n";
            return 2;
        }
    }

    namespace fs = std::filesystem;
    const fs::path outDir = opt.outDir;
    const fs::path manifestPath = outDir / (opt.outputStem + "_bucket_manifest.json");

    if (opt.finalizeExistingSplit)
    {
        std::error_code ec;
        fs::create_directories(outDir, ec);

        std::array<std::size_t, BucketCount> bucketCounts{};
        std::array<std::array<std::size_t, 33>, BucketCount> bucketPieceCounts{};
        std::string error;
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            const auto bucketCsv = outDir / (opt.outputStem + "_" + bucket_key(bucket) + ".csv");
            const auto bucketCountPath = fs::path(bucketCsv.string() + ".count");
            std::size_t count = 0;
            std::array<std::size_t, 33> pieceCounts{};
            if (fs::exists(bucketCountPath))
            {
                if (!read_count_cache(bucketCountPath, count, pieceCounts, error))
                {
                    std::cerr << error << "\n";
                    return 1;
                }
            }
            else
            {
                std::size_t totalSplitCount = 0;
                bool allSplitCountsExist = true;
                static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
                for (const auto& splitName : splitNames)
                {
                    const auto splitCountPath = fs::path(split_csv_path(outDir, opt.outputStem, splitName, bucket).string() + ".count");
                    if (!fs::exists(splitCountPath))
                    {
                        allSplitCountsExist = false;
                        break;
                    }
                    std::size_t splitCount = 0;
                    std::array<std::size_t, 33> splitPieceCounts{};
                    if (!read_count_cache(splitCountPath, splitCount, splitPieceCounts, error))
                    {
                        std::cerr << error << "\n";
                        return 1;
                    }
                    totalSplitCount += splitCount;
                    for (std::size_t i = 0; i < pieceCounts.size(); ++i)
                        pieceCounts[i] += splitPieceCounts[i];
                }
                if (!allSplitCountsExist)
                {
                    std::cerr << "Missing bucket count cache for " << bucket_key(bucket)
                              << " and no complete split outputs available.\n";
                    return 1;
                }
                count = totalSplitCount;
            }
            bucketCounts[bucket] = count;
            bucketPieceCounts[bucket] = pieceCounts;
        }

        SplitFinalizeResult splitResult;
        if (!finalize_existing_bucket_csvs(opt, outDir, bucketCounts, splitResult, error))
        {
            std::cerr << error << "\n";
            return 1;
        }

        std::ofstream manifest(manifestPath, std::ios::binary | std::ios::trunc);
        if (!manifest)
        {
            std::cerr << "Failed to open manifest for writing: " << manifestPath.string() << "\n";
            return 1;
        }

        const std::size_t totalRows = std::accumulate(bucketCounts.begin(), bucketCounts.end(), std::size_t{0});
        manifest << "{\n";
        manifest << "  \"finalize_existing_split\": true,\n";
        manifest << "  \"split_final\": true,\n";
        manifest << "  \"bucket_count\": " << BucketCount << ",\n";
        manifest << "  \"include_min_elo_column\": true,\n";
        manifest << "  \"include_source_column\": true,\n";
        manifest << "  \"csv_columns\": [\"fen\", \"psqt\", \"positional\", \"nnue\", \"min_elo\", \"source\"],\n";
        manifest << "  \"bucket_counts\": {\n";
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            manifest << "    \"" << bucket_key(bucket) << "\": " << bucketCounts[bucket];
            manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"bucket_piece_count_counts\": {\n";
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            manifest << "    \"" << bucket_key(bucket) << "\": {";
            bool first = true;
            for (std::size_t pieceCount = 0; pieceCount < bucketPieceCounts[bucket].size(); ++pieceCount)
            {
                if (bucketPieceCounts[bucket][pieceCount] == 0)
                    continue;
                manifest << (first ? "" : ", ");
                manifest << "\"" << pieceCount << "\": " << bucketPieceCounts[bucket][pieceCount];
                first = false;
            }
            manifest << "}";
            manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_config\": {\n";
        manifest << "    \"requested_train_ratio\": " << opt.trainRatio << ",\n";
        manifest << "    \"requested_validation_ratio\": " << opt.validationRatio << ",\n";
        manifest << "    \"requested_test_ratio\": " << opt.testRatio << ",\n";
        manifest << "    \"max_validation_rows\": " << opt.maxValidationRows << ",\n";
        manifest << "    \"max_test_rows\": " << opt.maxTestRows << ",\n";
        manifest << "    \"split_shards\": " << opt.splitShards << ",\n";
        manifest << "    \"split_workers\": " << opt.splitWorkers << "\n";
        manifest << "  },\n";
        manifest << "  \"split_totals\": {\n";
        manifest << "    \"total\": " << totalRows << ",\n";
        manifest << "    \"train\": " << splitResult.totals[0] << ",\n";
        manifest << "    \"validation\": " << splitResult.totals[1] << ",\n";
        manifest << "    \"test\": " << splitResult.totals[2] << "\n";
        manifest << "  },\n";
        manifest << "  \"split_bucket_paths\": {\n";
        static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": \"" << splitResult.paths[split][bucket].generic_string()
                         << "\"";
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_bucket_counts\": {\n";
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": " << splitResult.counts[split][bucket];
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_bucket_piece_count_counts\": {\n";
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": {";
                bool first = true;
                for (std::size_t pieceCount = 0; pieceCount < splitResult.pieceCounts[split][bucket].size(); ++pieceCount)
                {
                    if (splitResult.pieceCounts[split][bucket][pieceCount] == 0)
                        continue;
                    manifest << (first ? "" : ", ");
                    manifest << "\"" << pieceCount << "\": " << splitResult.pieceCounts[split][bucket][pieceCount];
                    first = false;
                }
                manifest << "}";
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_elapsed_sec\": " << splitResult.elapsedSec << ",\n";
        manifest << "  \"final_size_mb\": " << splitResult.finalSizeMb << "\n";
        manifest << "}\n";
        return 0;
    }

    using namespace Stockfish;
    namespace NN = Eval::NNUE;
    Bitboards::init();
    Position::init();
    const auto startedAt = std::chrono::steady_clock::now();

    auto networks = std::make_unique<NN::Networks>(
      NN::EvalFile{EvalFileDefaultNameBig, "None", ""},
      NN::EvalFile{EvalFileDefaultNameSmall, "None", ""});

    const std::string enginePath = argc > 0 ? argv[0] : std::string();
    const std::string binaryDir = CommandLine::get_binary_directory(argv[0]);
    const std::string evalPath = opt.evalFile.empty()
      ? std::string()
      : resolve_eval_file_path(opt.evalFile, binaryDir,
                               EvalFileDefaultNameBig
      );
    const std::string bigPath =
      !evalPath.empty() ? evalPath : resolve_eval_file_path(opt.bigEvalFile, binaryDir, EvalFileDefaultNameBig);
    const std::string smallPath =
      resolve_eval_file_path(opt.smallEvalFile, binaryDir, EvalFileDefaultNameSmall);

    if (!check_eval_file(bigPath, "Big NNUE")
        || !check_eval_file(smallPath, "Small NNUE")
    )
        return 1;

    networks->big.load(binaryDir, bigPath);
    networks->small.load(binaryDir, smallPath);
    auto onVerify = [](std::string_view msg) { std::cerr << msg; };
    networks->big.verify(bigPath, onVerify);
    networks->small.verify(smallPath, onVerify);

    std::istream* in = &std::cin;
    std::ifstream inFile;
    if (!opt.inPath.empty())
    {
        inFile.open(opt.inPath);
        if (!inFile)
        {
            std::cerr << "Failed to open input file: " << opt.inPath << "\n";
            return 1;
        }
        in = &inFile;
    }

    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec)
    {
        std::cerr << "Failed to create output directory: " << outDir.string() << "\n";
        return 1;
    }
    const fs::path bucketOutputDir = opt.bucketOutputDir.empty() ? outDir : fs::path(opt.bucketOutputDir);
    fs::create_directories(bucketOutputDir, ec);
    if (ec)
    {
        std::cerr << "Failed to create bucket output directory: " << bucketOutputDir.string() << "\n";
        return 1;
    }

    std::vector<fs::path> csvPaths(BucketCount);
    std::vector<fs::path> countPaths(BucketCount);
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        const std::string key = bucket_key(bucket);
        csvPaths[bucket] = bucketOutputDir / (opt.outputStem + "_" + key + (opt.streamOut ? ".fifo" : ".csv"));
        countPaths[bucket] = fs::path(csvPaths[bucket].string() + ".count");
        if (!opt.appendOut && !opt.splitFinal && !opt.streamOut)
        {
            fs::remove(csvPaths[bucket], ec);
            fs::remove(countPaths[bucket], ec);
        }
    }
    if (!opt.appendOut)
        fs::remove(manifestPath, ec);
    const fs::path shardRoot = split_shard_root(outDir, opt.outputStem);
    if (opt.splitFinal)
        fs::remove_all(shardRoot, ec);

    std::string error;
    std::array<FlatHashSet64, BucketCount> seen64;
    std::array<std::mutex, BucketCount>    seen64Mutexes;
    std::array<std::size_t, BucketCount> bucketCounts{};
    std::array<std::array<std::size_t, 33>, BucketCount> pieceCountCounts{};
    std::size_t excludeHashesLoaded = 0;
    std::size_t appendPreloadRows = 0;
    if (!load_exclude_hash_files(opt.excludeHashFiles, seen64, excludeHashesLoaded, error))
    {
        std::cerr << error << "\n";
        return 1;
    }

    if (opt.appendOut)
    {
        const auto appendPreloadStartedAt = std::chrono::steady_clock::now();
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            if (!fs::exists(csvPaths[bucket], ec) || ec)
                continue;
            const std::size_t before = bucketCounts[bucket];
            if (!load_hashes_from_csv(csvPaths[bucket], seen64[bucket], bucketCounts[bucket], pieceCountCounts[bucket], error))
            {
                std::cerr << error << "\n";
                return 1;
            }
            appendPreloadRows += bucketCounts[bucket] - before;
        }
        const double appendPreloadElapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - appendPreloadStartedAt).count();
        std::cout << "[append-preload] loaded " << format_grouped_u64(appendPreloadRows)
                  << " existing rows for dedup | elapsed " << std::fixed << std::setprecision(1)
                  << appendPreloadElapsed << "s\n";
        std::cout.flush();
    }
    SharedDedupState dedupState{&seen64, &seen64Mutexes, opt.dedup};

    std::vector<std::ofstream> outputs(BucketCount);
    if (!opt.splitFinal)
    {
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            ec.clear();
            const bool fileExists = fs::exists(csvPaths[bucket], ec) && !ec;
            ec.clear();
            const bool needHeader = opt.streamOut || !opt.appendOut || !fileExists
                                  || (fileExists && fs::file_size(csvPaths[bucket], ec) == 0);
            std::ios::openmode openMode = std::ios::binary | std::ios::out;
            if (!opt.streamOut)
                openMode |= opt.appendOut ? std::ios::app : std::ios::trunc;
            outputs[bucket].open(
              csvPaths[bucket],
              openMode);
            if (!outputs[bucket])
            {
                std::cerr << "Failed to open output CSV: " << csvPaths[bucket].string() << "\n";
                return 1;
            }
            if (needHeader)
                outputs[bucket] << output_csv_header(opt);
        }
    }
    std::array<std::vector<std::string>, BucketCount> shardBuffers;
    if (opt.splitFinal)
    {
        for (auto& perBucket : shardBuffers)
            perBucket.resize(std::max<std::size_t>(1, opt.splitShards));
    }

    GlobalStats stats;
    ProcessRuntimeProgress runtimeProgress;
    const auto processingStartedAt = std::chrono::steady_clock::now();
    stats.fensWritten = std::accumulate(bucketCounts.begin(), bucketCounts.end(), std::size_t{0});
    std::size_t lastFlushedFens = stats.fensWritten;
    bool targetReached = opt.minFens > 0 && stats.fensWritten >= opt.minFens;

    const std::size_t threadCount = std::max<std::size_t>(1, opt.threads);
    const std::size_t batchSize = std::max<std::size_t>(1, opt.batchGames);
    const std::size_t workQueueCapacity = std::max<std::size_t>(2, threadCount * 2);
    const std::size_t resultQueueCapacity = std::max<std::size_t>(1, threadCount);
    BoundedQueue<WorkItem> workQueue(workQueueCapacity);
    BoundedQueue<BatchResult> resultQueue(resultQueueCapacity);
    std::atomic<bool> stopRequested{targetReached};
    std::atomic<bool> failed{false};
    std::atomic<std::size_t> workersRemaining{threadCount};
    std::mutex errorMutex;
    std::string asyncError;
    std::size_t lineNo = 0;
    std::atomic<bool> progressMonitorStop{false};
    std::uint64_t readerReadUs = 0;
    std::uint64_t readerPushWaitUs = 0;
    std::size_t workItemsSubmitted = 0;

    auto set_error = [&](const std::string& message) {
        bool expected = false;
        if (failed.compare_exchange_strong(expected, true))
        {
            {
                std::lock_guard<std::mutex> lock(errorMutex);
                asyncError = message;
            }
            stopRequested.store(true);
            workQueue.close();
            resultQueue.close();
        }
    };

    std::thread readerThread([&]() {
        while (!stopRequested.load())
        {
            WorkItem item;
            std::size_t selectedFullBatch = 0;
            std::size_t selectedEndgameBatch = 0;
            std::string readErr;
            const auto readStartedAt = std::chrono::steady_clock::now();
            if (!read_games_chunk(*in, item.games, batchSize, lineNo, selectedFullBatch, selectedEndgameBatch, readErr))
            {
                set_error(readErr);
                return;
            }
            readerReadUs += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - readStartedAt).count());
            if (item.games.empty())
                break;
            item.sequence = workItemsSubmitted;
            const auto pushStartedAt = std::chrono::steady_clock::now();
            if (!workQueue.push(std::move(item)))
                break;
            readerPushWaitUs += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - pushStartedAt).count());
            workItemsSubmitted++;
        }
        workQueue.close();
    });

    std::vector<std::thread> computeThreads;
    computeThreads.reserve(threadCount);
    for (std::size_t workerIndex = 0; workerIndex < threadCount; ++workerIndex)
    {
        computeThreads.emplace_back([&]() {
            WorkerResources workerResources;
            WorkItem item;
            while (workQueue.pop(item))
            {
                if (failed.load())
                    break;
                BatchResult batch =
                  process_games_chunk(item.games, opt, *networks, workerResources, &dedupState);
                batch.sequence = item.sequence;
                if (!resultQueue.push(std::move(batch)))
                    break;
            }

            if (workersRemaining.fetch_sub(1) == 1)
                resultQueue.close();
        });
    }

    std::thread progressThread;
    if (!opt.splitFinal && !opt.finalizeExistingSplit && opt.progressEverySeconds > 0)
    {
        progressThread = std::thread([&]() {
            const auto progressStartedAt = processingStartedAt;
            auto nextReportAt = progressStartedAt + std::chrono::seconds(opt.progressEverySeconds);
            while (!progressMonitorStop.load())
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (progressMonitorStop.load())
                    break;
                const auto now = std::chrono::steady_clock::now();
                if (now < nextReportAt)
                    continue;
                nextReportAt += std::chrono::seconds(opt.progressEverySeconds);
                const auto scanned = runtimeProgress.positionsScanned.load(std::memory_order_relaxed);
                const auto written = runtimeProgress.positionsWritten.load(std::memory_order_relaxed);
                const auto duplicated = runtimeProgress.duplicateDiscarded.load(std::memory_order_relaxed);
                const auto discarded = scanned > written ? scanned - written : 0;
                const double elapsed =
                  std::chrono::duration<double>(now - progressStartedAt).count();
                const double rate = elapsed > 0.0 ? static_cast<double>(scanned) / elapsed : 0.0;
                std::cout << "[process-summary] " << opt.phaseLabel
                          << " | positions_scanned " << format_grouped_u64(scanned)
                          << " | positions_written " << format_grouped_u64(written)
                          << " | positions_discarded " << format_grouped_u64(discarded)
                          << " | duplicated_discarded " << format_grouped_u64(duplicated)
                          << " | rate " << format_grouped_u64(static_cast<std::uint64_t>(std::llround(rate))) << " positions/s"
                          << " | elapsed " << std::fixed << std::setprecision(1) << elapsed << "s\n";
                std::cout.flush();
            }
        });
    }

    std::map<std::size_t, BatchResult> pendingBatches;
    std::size_t nextSequenceToWrite = 0;
    bool activePhase = false;
    std::string activePhaseName;
    PhaseStats activePhaseStats;
    auto activePhaseStartedAt = processingStartedAt;

    auto update_active_phase = [&](const std::string& phaseName, const PhaseStats& phaseStats) {
        const auto now = std::chrono::steady_clock::now();
        if (!activePhase)
        {
            activePhase = true;
            activePhaseName = phaseName;
            activePhaseStats = {};
            activePhaseStartedAt = now;
            std::cout << "[phase " << activePhaseName << "] start\n";
            std::cout.flush();
        }
        else if (activePhaseName != phaseName)
        {
            const double elapsed = std::chrono::duration<double>(now - activePhaseStartedAt).count();
            print_phase_summary(activePhaseName, activePhaseStats, elapsed);
            activePhaseName = phaseName;
            activePhaseStats = {};
            activePhaseStartedAt = now;
            std::cout << "[phase " << activePhaseName << "] start\n";
            std::cout.flush();
        }
        add_phase_stats(activePhaseStats, phaseStats);
    };

    while (true)
    {
        BatchResult batch;
        const auto popStartedAt = std::chrono::steady_clock::now();
        if (!resultQueue.pop(batch))
            break;
        stats.resultPopWaitUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - popStartedAt).count());
        if (!batch.fatalError.empty())
        {
            set_error(batch.fatalError);
            break;
        }

        pendingBatches.emplace(batch.sequence, std::move(batch));
        while (true)
        {
            auto pendingIt = pendingBatches.find(nextSequenceToWrite);
            if (pendingIt == pendingBatches.end())
                break;
            BatchResult readyBatch = std::move(pendingIt->second);
            pendingBatches.erase(pendingIt);
            ++nextSequenceToWrite;

        const auto integrateStartedAt = std::chrono::steady_clock::now();
        stats.gamesSelected += readyBatch.stats.gamesProcessed;
        stats.gamesSelectedFull += readyBatch.stats.gamesFull;
        stats.gamesSelectedEndgame += readyBatch.stats.gamesEndgameOnly;
        stats.gamesProcessed += readyBatch.stats.gamesProcessed;
        stats.gamesFull += readyBatch.stats.gamesFull;
        stats.gamesEndgameOnly += readyBatch.stats.gamesEndgameOnly;
        stats.gamesSkippedNoMoves += readyBatch.stats.gamesSkippedNoMoves;
        stats.gamesSkippedInvalid += readyBatch.stats.gamesSkippedInvalid;
        stats.positionsScanned += readyBatch.stats.positionsScanned;
        stats.positionsSelected += readyBatch.stats.positionsSelected;
        stats.duplicateFensSkipped += readyBatch.stats.duplicateSkipped;
        for (const auto& [phaseName, phaseBatchStats] : readyBatch.stats.phaseStats)
        {
            auto& phaseStats = stats.phaseStats[phaseName];
            phaseStats.positionsScanned += phaseBatchStats.positionsScanned;
            phaseStats.positionsWritten += phaseBatchStats.positionsWritten;
            phaseStats.duplicateSkipped += phaseBatchStats.duplicateSkipped;
        }
        stats.batchSetupUs += readyBatch.stats.batchSetupUs;
        stats.batchProcessUs += readyBatch.stats.batchProcessUs;
        stats.gameSetupUs += readyBatch.stats.gameSetupUs;
        stats.moveParseUs += readyBatch.stats.moveParseUs;
        stats.moveApplyUs += readyBatch.stats.moveApplyUs;
        stats.fenSerializeUs += readyBatch.stats.fenSerializeUs;
        stats.dedupUs += readyBatch.stats.dedupUs;
        stats.rowBuildUs += readyBatch.stats.rowBuildUs;
        stats.batchesCompleted++;
        runtimeProgress.positionsScanned.fetch_add(readyBatch.stats.positionsScanned, std::memory_order_relaxed);
        runtimeProgress.duplicateDiscarded.fetch_add(readyBatch.stats.duplicateSkipped, std::memory_order_relaxed);
        stats.resultIntegrateUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - integrateStartedAt).count());

        const auto writeStartedAt = std::chrono::steady_clock::now();
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            for (const RowData& row : readyBatch.rowsByBucket[bucket])
            {
                std::uint64_t rowHash = 0;
                rowHash = hash_row64(row);

                if (opt.splitFinal)
                {
                    const std::size_t shardId = rowHash % std::max<std::size_t>(1, opt.splitShards);
                    std::ostringstream oss;
                    oss << std::hex << std::setw(16) << std::setfill('0') << rowHash << std::dec
                        << '\t' << row.pieceCount
                        << '\t' << row.fen
                        << '\t' << int(row.psqt)
                        << '\t' << int(row.positional)
                        << '\t' << int(row.nnue);
                    oss << '\t';
                    if (row.minElo)
                        oss << *row.minElo;
                    oss << '\t' << row.source;
                    oss << '\n';
                    shardBuffers[bucket][shardId] += oss.str();
                    if (shardBuffers[bucket][shardId].size() >= 64 * 1024)
                    {
                        if (!append_text_file(split_shard_path(shardRoot, bucket, shardId),
                                              shardBuffers[bucket][shardId],
                                              error))
                        {
                            set_error(error);
                            break;
                        }
                        shardBuffers[bucket][shardId].clear();
                    }
                }
                else
                {
                    outputs[bucket] << '"' << row.fen << '"' << ','
                                    << int(row.psqt) << ','
                                    << int(row.positional) << ','
                                    << int(row.nnue);
                    outputs[bucket] << ',';
                    if (row.minElo)
                        outputs[bucket] << *row.minElo;
                    outputs[bucket] << ',' << row.source;
                    outputs[bucket] << '\n';
                    if (!outputs[bucket])
                    {
                        set_error("Failed writing output CSV: " + csvPaths[bucket].string());
                        break;
                    }
                }
                bucketCounts[bucket]++;
                if (row.pieceCount >= 0 && row.pieceCount < static_cast<int>(pieceCountCounts[bucket].size()))
                    pieceCountCounts[bucket][static_cast<std::size_t>(row.pieceCount)]++;
                stats.fensWritten++;
                runtimeProgress.positionsWritten.fetch_add(1, std::memory_order_relaxed);
                if (bucket < 2)
                    runtimeProgress.endgameWritten.fetch_add(1, std::memory_order_relaxed);
            }
            if (failed.load())
                break;
        }

        if (failed.load())
            break;
        stats.outputWriteUs += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - writeStartedAt).count());

        for (const auto& [phaseName, phaseStats] : readyBatch.stats.phaseSegments)
            update_active_phase(phaseName, phaseStats);

        if (opt.flushEveryFens > 0 && stats.fensWritten - lastFlushedFens >= opt.flushEveryFens)
        {
            const auto flushStartedAt = std::chrono::steady_clock::now();
            const bool flushOk =
              opt.splitFinal
                ? flush_shard_buffers(shardRoot, shardBuffers, error)
                : flush_outputs(outputs, countPaths, bucketCounts, pieceCountCounts, !opt.streamOut, error);
            if (!flushOk)
            {
                set_error(error);
                break;
            }
            if (opt.splitFinal)
                clear_shard_buffers(shardBuffers);
            lastFlushedFens = stats.fensWritten;
            stats.flushUs += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - flushStartedAt).count());
        }

        if (opt.progressEveryGames > 0 && stats.gamesSelected % opt.progressEveryGames == 0)
        {
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - processingStartedAt).count();
            const double rate = elapsed > 0.0 ? static_cast<double>(stats.gamesSelected) / elapsed : 0.0;
            std::cout << "Process stream | selected " << stats.gamesSelected
                      << " games | full " << stats.gamesSelectedFull
                      << " | endgame_only " << stats.gamesSelectedEndgame
                      << " | positions_written " << stats.fensWritten
                      << " | rate " << std::fixed << std::setprecision(1) << rate << " games/s\n";
        }

        if (opt.minFens > 0 && stats.fensWritten >= opt.minFens && !targetReached)
        {
            targetReached = true;
            stopRequested.store(true);
            workQueue.close();
        }

            if (failed.load())
                break;
        }

        if (failed.load())
            break;
    }

    stopRequested.store(true);
    workQueue.close();
    resultQueue.close();
    readerThread.join();
    stats.readerReadUs = readerReadUs;
    stats.readerPushWaitUs = readerPushWaitUs;
    stats.workItemsSubmitted = workItemsSubmitted;
    for (auto& thread : computeThreads)
        thread.join();
    progressMonitorStop.store(true);
    if (progressThread.joinable())
        progressThread.join();

    if (failed.load())
    {
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!asyncError.empty())
                error = asyncError;
        }
        std::cerr << (error.empty() ? std::string("Asynchronous pipeline failure") : error) << "\n";
        return 1;
    }

    if (activePhase)
    {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - activePhaseStartedAt).count();
        print_phase_summary(activePhaseName, activePhaseStats, elapsed);
    }

    const auto finalFlushStartedAt = std::chrono::steady_clock::now();
    const bool finalFlushOk =
      opt.splitFinal
        ? flush_shard_buffers(shardRoot, shardBuffers, error)
        : flush_outputs(outputs, countPaths, bucketCounts, pieceCountCounts, !opt.streamOut, error);
    const auto finalFlushEndedAt = std::chrono::steady_clock::now();
    if (!finalFlushOk)
    {
        std::cerr << error << "\n";
        return 1;
    }
    stats.flushUs += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(finalFlushEndedAt - finalFlushStartedAt).count());
    if (opt.splitFinal)
        clear_shard_buffers(shardBuffers);
    else
    {
        for (auto& out : outputs)
            out.close();
    }

    SplitFinalizeResult splitResult;
    if (opt.splitFinal)
    {
        if (!finalize_split_outputs(opt, outDir, shardRoot, bucketCounts, splitResult, error))
        {
            std::cerr << error << "\n";
            return 1;
        }
    }

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    const double sizeMb = opt.splitFinal ? splitResult.finalSizeMb : total_size_mb(csvPaths);

    std::ofstream manifest(manifestPath, std::ios::binary | std::ios::trunc);
    if (!manifest)
    {
        std::cerr << "Failed to open manifest for writing: " << manifestPath.string() << "\n";
        return 1;
    }

    manifest << "{\n";
    manifest << "  \"selection_rule\": \"all games contribute endgames (b00/b01); "
                "games marked F contribute full games across all 16 buckets\",\n";
    manifest << "  \"min_fens_target\": " << opt.minFens << ",\n";
    manifest << "  \"bucket_count\": " << BucketCount << ",\n";
    manifest << "  \"dedup_fens\": " << (opt.dedup ? "true" : "false") << ",\n";
    manifest << "  \"dedup_hash_bits\": 64,\n";
    manifest << "  \"append_out\": " << (opt.appendOut ? "true" : "false") << ",\n";
    manifest << "  \"stream_out\": " << (opt.streamOut ? "true" : "false") << ",\n";
    manifest << "  \"exclude_hash_files\": [";
    for (std::size_t i = 0; i < opt.excludeHashFiles.size(); ++i)
    {
        manifest << (i == 0 ? "" : ", ") << "\"" << opt.excludeHashFiles[i] << "\"";
    }
    manifest << "],\n";
    manifest << "  \"exclude_hashes_loaded\": " << excludeHashesLoaded << ",\n";
    manifest << "  \"csv_columns\": [\"fen\", \"psqt\", \"positional\", \"nnue\", \"min_elo\", \"source\"],\n";
    manifest << "  \"big_eval_file\": \"" << bigPath << "\",\n";
    manifest << "  \"small_eval_file\": \"" << smallPath << "\",\n";
    manifest << "  \"depth\": 0,\n";
    manifest << "  \"include_min_elo_column\": true,\n";
    manifest << "  \"include_source_column\": true,\n";
    manifest << "  \"threads\": " << threadCount << ",\n";
    manifest << "  \"batch_games\": " << batchSize << ",\n";
    manifest << "  \"target_reached\": " << (targetReached ? "true" : "false") << ",\n";
    manifest << "  \"elapsed_sec\": " << elapsed << ",\n";
    manifest << "  \"size_mb\": " << sizeMb << ",\n";
    manifest << "  \"split_final\": " << (opt.splitFinal ? "true" : "false") << ",\n";
    manifest << "  \"bucket_counts\": {\n";
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        manifest << "    \"" << bucket_key(bucket) << "\": " << bucketCounts[bucket];
        manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
    }
    manifest << "  },\n";
    manifest << "  \"bucket_piece_count_counts\": {\n";
    for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
    {
        manifest << "    \"" << bucket_key(bucket) << "\": {";
        bool first = true;
        for (std::size_t pieceCount = 0; pieceCount < pieceCountCounts[bucket].size(); ++pieceCount)
        {
            if (pieceCountCounts[bucket][pieceCount] == 0)
                continue;
            manifest << (first ? "" : ", ");
            manifest << "\"" << pieceCount << "\": " << pieceCountCounts[bucket][pieceCount];
            first = false;
        }
        manifest << "}";
        manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
    }
    manifest << "  }";
    if (!opt.splitFinal)
    {
        manifest << ",\n";
        manifest << "  \"bucket_paths\": {\n";
        for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
        {
            manifest << "    \"" << bucket_key(bucket) << "\": \"" << csvPaths[bucket].generic_string() << "\"";
            manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
        }
        manifest << "  },\n";
    }
    else
    {
        const std::size_t totalRows = std::accumulate(bucketCounts.begin(), bucketCounts.end(), std::size_t{0});
        manifest << ",\n";
        manifest << "  \"temporary_bucket_paths_deleted\": true,\n";
        manifest << "  \"split_config\": {\n";
        manifest << "    \"requested_train_ratio\": " << opt.trainRatio << ",\n";
        manifest << "    \"requested_validation_ratio\": " << opt.validationRatio << ",\n";
        manifest << "    \"requested_test_ratio\": " << opt.testRatio << ",\n";
        manifest << "    \"max_validation_rows\": " << opt.maxValidationRows << ",\n";
        manifest << "    \"max_test_rows\": " << opt.maxTestRows << ",\n";
        manifest << "    \"split_shards\": " << opt.splitShards << "\n";
        manifest << "  },\n";
        manifest << "  \"split_totals\": {\n";
        manifest << "    \"total\": " << totalRows << ",\n";
        manifest << "    \"train\": " << splitResult.totals[0] << ",\n";
        manifest << "    \"validation\": " << splitResult.totals[1] << ",\n";
        manifest << "    \"test\": " << splitResult.totals[2] << "\n";
        manifest << "  },\n";
        manifest << "  \"split_bucket_paths\": {\n";
        static const std::array<std::string, 3> splitNames = {"train", "validation", "test"};
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": \""
                         << splitResult.paths[split][bucket].generic_string() << "\"";
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_bucket_counts\": {\n";
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": "
                         << splitResult.counts[split][bucket];
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_bucket_piece_count_counts\": {\n";
        for (std::size_t split = 0; split < splitNames.size(); ++split)
        {
            manifest << "    \"" << splitNames[split] << "\": {\n";
            for (std::size_t bucket = 0; bucket < BucketCount; ++bucket)
            {
                manifest << "      \"" << bucket_key(bucket) << "\": {";
                bool first = true;
                for (std::size_t pieceCount = 0; pieceCount < splitResult.pieceCounts[split][bucket].size(); ++pieceCount)
                {
                    if (splitResult.pieceCounts[split][bucket][pieceCount] == 0)
                        continue;
                    manifest << (first ? "" : ", ");
                    manifest << "\"" << pieceCount << "\": "
                             << splitResult.pieceCounts[split][bucket][pieceCount];
                    first = false;
                }
                manifest << "}";
                manifest << (bucket + 1 == BucketCount ? "\n" : ",\n");
            }
            manifest << "    }";
            manifest << (split + 1 == splitNames.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"split_elapsed_sec\": " << splitResult.elapsedSec << ",\n";
        manifest << "  \"final_size_mb\": " << splitResult.finalSizeMb << ",\n";
    }
    manifest << "  \"helper_phase_stats\": {\n";
    {
        bool firstPhase = true;
        for (const auto& [phaseName, phaseStats] : stats.phaseStats)
        {
            manifest << (firstPhase ? "" : ",\n");
            firstPhase = false;
            manifest << "    \"" << json_escape_string(phaseName) << "\": {"
                     << "\"positions_scanned\": " << phaseStats.positionsScanned
                     << ", \"positions_written\": " << phaseStats.positionsWritten
                     << ", \"duplicate_fens_skipped\": " << phaseStats.duplicateSkipped
                     << "}";
        }
        manifest << "\n";
    }
    manifest << "  },\n";
    manifest << "  \"stats\": {\n";
    manifest << "    \"games_selected\": " << stats.gamesSelected << ",\n";
    manifest << "    \"games_selected_full\": " << stats.gamesSelectedFull << ",\n";
    manifest << "    \"games_selected_endgame_only\": " << stats.gamesSelectedEndgame << ",\n";
    manifest << "    \"games_processed\": " << stats.gamesProcessed << ",\n";
    manifest << "    \"games_full\": " << stats.gamesFull << ",\n";
    manifest << "    \"games_endgame_only\": " << stats.gamesEndgameOnly << ",\n";
    manifest << "    \"games_skipped_no_moves\": " << stats.gamesSkippedNoMoves << ",\n";
    manifest << "    \"games_skipped_invalid_move\": " << stats.gamesSkippedInvalid << ",\n";
    manifest << "    \"positions_scanned\": " << stats.positionsScanned << ",\n";
    manifest << "    \"positions_selected\": " << stats.positionsSelected << ",\n";
    manifest << "    \"duplicate_fens_skipped\": " << stats.duplicateFensSkipped << ",\n";
    manifest << "    \"fens_written\": " << stats.fensWritten << ",\n";
    manifest << "    \"batches_completed\": " << stats.batchesCompleted << ",\n";
    manifest << "    \"work_items_submitted\": " << stats.workItemsSubmitted << ",\n";
    manifest << "    \"batch_setup_us\": " << stats.batchSetupUs << ",\n";
    manifest << "    \"batch_process_us\": " << stats.batchProcessUs << ",\n";
    manifest << "    \"reader_read_us\": " << stats.readerReadUs << ",\n";
    manifest << "    \"reader_push_wait_us\": " << stats.readerPushWaitUs << ",\n";
    manifest << "    \"result_pop_wait_us\": " << stats.resultPopWaitUs << ",\n";
    manifest << "    \"result_integrate_us\": " << stats.resultIntegrateUs << ",\n";
    manifest << "    \"output_write_us\": " << stats.outputWriteUs << ",\n";
    manifest << "    \"flush_us\": " << stats.flushUs << ",\n";
    manifest << "    \"game_setup_us\": " << stats.gameSetupUs << ",\n";
    manifest << "    \"move_parse_us\": " << stats.moveParseUs << ",\n";
    manifest << "    \"move_apply_us\": " << stats.moveApplyUs << ",\n";
    manifest << "    \"fen_serialize_us\": " << stats.fenSerializeUs << ",\n";
    manifest << "    \"dedup_us\": " << stats.dedupUs << ",\n";
    manifest << "    \"row_build_us\": " << stats.rowBuildUs << "\n";
    manifest << "  }\n";
    manifest << "}\n";

    return 0;
}

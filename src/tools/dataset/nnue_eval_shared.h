#pragma once

#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "engine.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/network.h"
#include "position.h"
#include "search.h"
#include "types.h"
#include "uci.h"

namespace nnue_eval_shared {

enum class OutputNet {
    Big,
    Small
};

enum class NnueLabelMode {
    Adjusted,
    Raw
};

struct EvalOptions {
    OutputNet   outputNet      = OutputNet::Big;
    bool        unscaledOutput = false;
    NnueLabelMode nnueLabelMode = NnueLabelMode::Adjusted;
    int         searchDepth    = 0;
    std::size_t searchHashMb   = 16;
    std::size_t searchThreads  = 1;
    bool        evaluateTerminalPv = false;
};

struct EvalTriplet {
    Stockfish::Value psqt       = Stockfish::VALUE_ZERO;
    Stockfish::Value positional = Stockfish::VALUE_ZERO;
    Stockfish::Value nnue       = Stockfish::VALUE_ZERO;
};

struct SearchOutcome {
    int         depth = 0;
    std::string bestMove;
    std::string pv;
    std::vector<Stockfish::Move> pvMoves;
    Stockfish::Value rootScore = Stockfish::VALUE_NONE;
    bool             rootScoreIsDecisive = false;
};

struct SearchResources {
    struct SharedCapture {
        std::mutex  mtx;
        int         depth = 0;
        bool        hadNoMoves = false;
        bool        capturePv = false;
        std::string bestMove;
        std::string pv;
    };

    std::unique_ptr<Stockfish::Engine> engine;
    std::unique_ptr<Stockfish::Eval::NNUE::AccumulatorCaches> pvCaches;
    std::unique_ptr<Stockfish::Eval::NNUE::AccumulatorStack>  pvAccumulators;
    std::shared_ptr<SharedCapture>                            capture;
};

struct SearchTimingStats {
    std::uint64_t tracedPositions = 0;
    std::uint64_t rootEvalPositions = 0;
    std::uint64_t rootEvalUs = 0;
    std::uint64_t rootFenSerializeUs = 0;
    std::uint64_t engineSetPositionUs = 0;
    std::uint64_t engineGoWaitUs = 0;
    std::uint64_t pvReplayUs = 0;
    std::uint64_t pvEvalUs = 0;
    std::uint64_t totalDepthPathUs = 0;

    void merge_from(const SearchTimingStats& other) {
        tracedPositions += other.tracedPositions;
        rootEvalPositions += other.rootEvalPositions;
        rootEvalUs += other.rootEvalUs;
        rootFenSerializeUs += other.rootFenSerializeUs;
        engineSetPositionUs += other.engineSetPositionUs;
        engineGoWaitUs += other.engineGoWaitUs;
        pvReplayUs += other.pvReplayUs;
        pvEvalUs += other.pvEvalUs;
        totalDepthPathUs += other.totalDepthPathUs;
    }
};

struct EvaluatedPosition {
    EvalTriplet               root;
    Stockfish::Value          searchValueRaw = Stockfish::VALUE_NONE;
    bool                      searchValueIsDecisive = false;
    bool                      pvNeedsSignFlip = false;
    std::optional<EvalTriplet> pv;
};

using SearchClock = std::chrono::steady_clock;

inline std::uint64_t elapsed_us(const SearchClock::time_point start,
                                const SearchClock::time_point end) {
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

inline Stockfish::Value mix_nnue(Stockfish::Value psqt, Stockfish::Value positional) {
    return Stockfish::Value((125LL * int(psqt) + 131LL * int(positional)) / 128);
}

inline constexpr int nnue_complexity_divisor() {
    return 18236;
}

inline Stockfish::Value mix_adjusted_nnue_from_scaled(Stockfish::Value psqtScaled,
                                                      Stockfish::Value positionalScaled) {
    int nnue = int((125LL * int(psqtScaled) + 131LL * int(positionalScaled)) / 128);
    const int complexity = std::abs(int(psqtScaled) - int(positionalScaled));
    nnue -= int((1LL * nnue * complexity) / nnue_complexity_divisor());
    return Stockfish::Value(nnue);
}

inline std::tuple<Stockfish::Value, Stockfish::Value>
scaled_components_for_label(const EvalOptions& opt,
                            Stockfish::Value   psqt,
                            Stockfish::Value   positional) {
    using namespace Stockfish::Eval::NNUE;

    if (!opt.unscaledOutput)
        return std::make_tuple(psqt, positional);

    return std::make_tuple(Stockfish::Value(int(psqt) / OutputScale),
                           Stockfish::Value(int(positional) / OutputScale));
}

inline Stockfish::Value make_nnue_label(const EvalOptions& opt,
                                        Stockfish::Value   psqt,
                                        Stockfish::Value   positional) {
    auto [psqtScaled, positionalScaled] = scaled_components_for_label(opt, psqt, positional);
    if (opt.nnueLabelMode == NnueLabelMode::Raw)
        return mix_nnue(psqtScaled, positionalScaled);
    return mix_adjusted_nnue_from_scaled(psqtScaled, positionalScaled);
}

inline std::tuple<Stockfish::Value, Stockfish::Value>
evaluate_selected_network(const EvalOptions&                     opt,
                          Stockfish::Position&                   pos,
                          Stockfish::Eval::NNUE::Networks&       networks,
                          Stockfish::Eval::NNUE::AccumulatorCaches& caches,
                          Stockfish::Eval::NNUE::AccumulatorStack&  accumulators) {
    auto scale_output = [&](Stockfish::Value psqt, Stockfish::Value positional) {
        if (!opt.unscaledOutput)
            return std::make_tuple(psqt, positional);

        using namespace Stockfish::Eval::NNUE;
        return std::make_tuple(Stockfish::Value((1LL * int(psqt) * OutputScale)),
                               Stockfish::Value((1LL * int(positional) * OutputScale)));
    };

    if (opt.outputNet == OutputNet::Small)
    {
        auto [psqt, positional] = networks.small.evaluate(pos, accumulators, caches.small);
        return scale_output(psqt, positional);
    }

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, caches.big);
    return scale_output(psqt, positional);
}

inline void set_engine_option(Stockfish::Engine& engine,
                              const std::string& name,
                              const std::string& value) {
    std::istringstream is("name " + name + " value " + value);
    engine.get_options().setoption(is);
}

inline std::vector<std::string> split_space_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream       iss(text);
    std::string              token;
    while (iss >> token)
        tokens.push_back(token);
    return tokens;
}

inline bool replay_pv_to_position(const std::string&              rootFen,
                                  const std::vector<Stockfish::Move>& pvMoves,
                                  Stockfish::Position&             finalPos,
                                  std::string&                     error) {
    using namespace Stockfish;

    std::vector<StateInfo> states(pvMoves.size() + 1);
    finalPos.set(rootFen, false, &states[0]);

    // Stockfish represents "no legal move" root results with a single Move::none().
    // For dataset export, treat that as an empty PV and keep the root position.
    if (pvMoves.size() == 1 && pvMoves[0] == Move::none())
        return true;

    for (std::size_t i = 0; i < pvMoves.size(); ++i)
    {
        const Move move = pvMoves[i];
        if (move == Move::none())
        {
            error = "Illegal PV move at ply " + std::to_string(i + 1)
                  + " for root FEN: " + rootFen
                  + " (pv length " + std::to_string(pvMoves.size()) + ")";
            return false;
        }

        finalPos.do_move(move, states[i + 1]);
    }

    return true;
}

inline bool replay_pv_to_position_from_text(const std::string&  rootFen,
                                            const std::string&  pv,
                                            Stockfish::Position& finalPos,
                                            std::string&        error) {
    using namespace Stockfish;

    const auto moves = split_space_tokens(pv);
    std::vector<StateInfo> states(moves.size() + 1);
    finalPos.set(rootFen, false, &states[0]);

    for (std::size_t i = 0; i < moves.size(); ++i)
    {
        const Move move = UCIEngine::to_move(finalPos, moves[i]);
        if (move == Move::none())
        {
            error = "Illegal PV move at ply " + std::to_string(i + 1) + ": " + moves[i]
                  + " for root FEN: " + rootFen
                  + " (pv text length " + std::to_string(moves.size()) + ")";
            return false;
        }

        finalPos.do_move(move, states[i + 1]);
    }

    return true;
}

inline bool search_position(const EvalOptions& opt,
                            SearchResources&   resources,
                            const std::string& rootFen,
                            SearchTimingStats* timing,
                            SearchOutcome&     outcome,
                            std::string&       error) {
    using namespace Stockfish;
    if (!resources.engine || !resources.capture)
    {
        error = "Search resources not initialized";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(resources.capture->mtx);
        resources.capture->depth = 0;
        resources.capture->hadNoMoves = false;
        if (opt.evaluateTerminalPv)
        {
            resources.capture->bestMove.clear();
            resources.capture->pv.clear();
        }
    }

    SearchClock::time_point start;
    if (timing)
        start = SearchClock::now();
    resources.engine->set_position(rootFen, {});
    if (timing)
        timing->engineSetPositionUs += elapsed_us(start, SearchClock::now());

    Search::LimitsType limits;
    limits.depth = Stockfish::Depth(opt.searchDepth);
    limits.startTime = Stockfish::now();

    if (timing)
        start = SearchClock::now();
    resources.engine->go(limits);
    resources.engine->wait_for_search_finished();
    if (timing)
        timing->engineGoWaitUs += elapsed_us(start, SearchClock::now());

    bool hadNoMoves = false;
    {
        std::lock_guard<std::mutex> lock(resources.capture->mtx);
        outcome.depth = resources.capture->depth;
        hadNoMoves = resources.capture->hadNoMoves;
        if (opt.evaluateTerminalPv)
        {
            outcome.bestMove = resources.capture->bestMove;
            outcome.pv = resources.capture->pv;
        }

        if (hadNoMoves)
            outcome.pv.clear();
    }
    outcome.rootScore = resources.engine->get_last_root_score();
    outcome.rootScoreIsDecisive =
      outcome.rootScore != Stockfish::VALUE_NONE && Stockfish::is_decisive(outcome.rootScore);
    if (outcome.rootScore == Stockfish::VALUE_NONE)
    {
        error = "Search finished without root score information";
        return false;
    }

    if (opt.evaluateTerminalPv)
    {
        outcome.pvMoves = resources.engine->get_last_pv_moves();

        if (outcome.pvMoves.size() == 1 && outcome.pvMoves[0] == Stockfish::Move::none())
            outcome.pvMoves.clear();
    }

    if (opt.evaluateTerminalPv && outcome.bestMove == "(none)")
        outcome.bestMove.clear();

    if (opt.evaluateTerminalPv && !hadNoMoves && outcome.depth == 0 && outcome.pvMoves.empty()
        && outcome.pv.empty() && outcome.bestMove.empty() && outcome.rootScore == Stockfish::VALUE_NONE)
    {
        error = "Search finished without PV information";
        return false;
    }

    if (opt.evaluateTerminalPv && outcome.pv.empty() && !outcome.bestMove.empty())
        outcome.pv = outcome.bestMove;

    return true;
}

inline bool initialize_search_resources(const EvalOptions&               opt,
                                        const std::string&               enginePath,
                                        const std::string&               bigPath,
                                        const std::string&               smallPath,
                                        Stockfish::Eval::NNUE::Networks& networks,
                                        SearchResources&                 resources,
                                        std::string&                     error) {
    (void) error;
    if (opt.searchDepth <= 0)
        return true;

    Stockfish::Engine::StartupOptions startupOptions;
    startupOptions.threads = opt.searchThreads;
    startupOptions.hashMb = opt.searchHashMb;
    startupOptions.multiPV = 1;
    startupOptions.evalFile = bigPath;
    startupOptions.evalFileSmall = smallPath;
    resources.engine = std::make_unique<Stockfish::Engine>(enginePath, std::move(startupOptions));
    resources.capture = std::make_shared<SearchResources::SharedCapture>();
    resources.capture->capturePv = opt.evaluateTerminalPv;
    resources.engine->set_on_verify_networks([](std::string_view) {});
    resources.engine->set_on_update_no_moves([capture = resources.capture](const Stockfish::Engine::InfoShort&) {
        std::lock_guard<std::mutex> lock(capture->mtx);
        capture->hadNoMoves = true;
        capture->depth = 0;
        capture->bestMove.clear();
        capture->pv.clear();
    });
    resources.engine->set_on_iter([](const Stockfish::Engine::InfoIter&) {});
    auto capture = resources.capture;
    resources.engine->set_on_update_full([capture](const Stockfish::Engine::InfoFull& info) {
        if (info.multiPV != 1)
            return;

        std::lock_guard<std::mutex> lock(capture->mtx);
        if (info.depth >= capture->depth)
        {
            capture->depth = info.depth;
            if (capture->capturePv)
                capture->pv = std::string(info.pv);
        }
    });
    if (opt.evaluateTerminalPv)
    {
        resources.engine->set_on_bestmove([capture](std::string_view bestMove, std::string_view) {
            std::lock_guard<std::mutex> lock(capture->mtx);
            capture->bestMove = std::string(bestMove);
        });
        resources.pvCaches = std::make_unique<Stockfish::Eval::NNUE::AccumulatorCaches>(networks);
        resources.pvAccumulators = std::make_unique<Stockfish::Eval::NNUE::AccumulatorStack>();
    }
    else
    {
        resources.engine->set_on_bestmove([](std::string_view, std::string_view) {});
    }
    return true;
}

inline bool evaluate_position_with_optional_pv(
  const EvalOptions&                          opt,
  Stockfish::Position&                        pos,
  Stockfish::Eval::NNUE::Networks&            networks,
  Stockfish::Eval::NNUE::AccumulatorCaches&   caches,
  Stockfish::Eval::NNUE::AccumulatorStack&    accumulators,
  SearchResources*                            resources,
  SearchTimingStats*                          timing,
  EvaluatedPosition&                          evaluated,
  std::string&                                error) {
    SearchClock::time_point totalStart;
    SearchClock::time_point stageStart;
    if (timing && opt.searchDepth > 0)
        totalStart = SearchClock::now();
    if (timing)
        stageStart = SearchClock::now();
    auto [psqt, positional] = evaluate_selected_network(opt, pos, networks, caches, accumulators);
    if (timing)
    {
        timing->rootEvalPositions++;
        timing->rootEvalUs += elapsed_us(stageStart, SearchClock::now());
    }
    evaluated.root.psqt = psqt;
    evaluated.root.positional = positional;
    evaluated.root.nnue = make_nnue_label(opt, psqt, positional);
    evaluated.pv.reset();

    if (opt.searchDepth <= 0)
        return true;
    if (resources == nullptr || !resources->engine
        || (opt.evaluateTerminalPv && (!resources->pvCaches || !resources->pvAccumulators)))
    {
        error = "Search resources not initialized";
        return false;
    }

    SearchOutcome search;
    if (timing)
        stageStart = SearchClock::now();
    const std::string rootFen = pos.fen();
    if (timing)
        timing->rootFenSerializeUs += elapsed_us(stageStart, SearchClock::now());
    if (!search_position(opt, *resources, rootFen, timing, search, error))
        return false;
    evaluated.searchValueRaw = search.rootScore;
    evaluated.searchValueIsDecisive = search.rootScoreIsDecisive;

    if (!opt.evaluateTerminalPv)
    {
        if (timing)
        {
            timing->tracedPositions++;
            timing->totalDepthPathUs += elapsed_us(totalStart, SearchClock::now());
        }
        return true;
    }

    Stockfish::Position pvPos;
    if (timing)
        stageStart = SearchClock::now();
    if (!search.pvMoves.empty())
    {
        if (!replay_pv_to_position(rootFen, search.pvMoves, pvPos, error))
            return false;
    }
    else
    {
        if (!replay_pv_to_position_from_text(rootFen, search.pv, pvPos, error))
            return false;
    }
    if (timing)
        timing->pvReplayUs += elapsed_us(stageStart, SearchClock::now());

    resources->pvAccumulators->reset();
    if (timing)
        stageStart = SearchClock::now();
    auto [pvPsqt, pvPositional] = evaluate_selected_network(
      opt, pvPos, networks, *resources->pvCaches, *resources->pvAccumulators);
    if (timing)
        timing->pvEvalUs += elapsed_us(stageStart, SearchClock::now());
    evaluated.pvNeedsSignFlip = pvPos.side_to_move() != pos.side_to_move();
    evaluated.pv = EvalTriplet{
      pvPsqt,
      pvPositional,
      make_nnue_label(opt, pvPsqt, pvPositional)};
    if (timing)
    {
        timing->tracedPositions++;
        timing->totalDepthPathUs += elapsed_us(totalStart, SearchClock::now());
    }
    return true;
}

}  // namespace nnue_eval_shared

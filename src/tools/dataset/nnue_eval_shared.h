#pragma once

#include <cstdlib>
#include <tuple>

#include "nnue/nnue_accumulator.h"
#include "nnue/network.h"
#include "position.h"
#include "types.h"

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
    OutputNet     outputNet      = OutputNet::Big;
    bool          unscaledOutput = false;
    NnueLabelMode nnueLabelMode  = NnueLabelMode::Adjusted;
};

struct EvalTriplet {
    Stockfish::Value psqt       = Stockfish::VALUE_ZERO;
    Stockfish::Value positional = Stockfish::VALUE_ZERO;
    Stockfish::Value nnue       = Stockfish::VALUE_ZERO;
};

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
evaluate_selected_network(const EvalOptions&                        opt,
                          Stockfish::Position&                      pos,
                          Stockfish::Eval::NNUE::Networks&          networks,
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

inline EvalTriplet evaluate_position(const EvalOptions&                        opt,
                                     Stockfish::Position&                      pos,
                                     Stockfish::Eval::NNUE::Networks&          networks,
                                     Stockfish::Eval::NNUE::AccumulatorCaches& caches,
                                     Stockfish::Eval::NNUE::AccumulatorStack&  accumulators) {
    auto [psqt, positional] = evaluate_selected_network(opt, pos, networks, caches, accumulators);
    return EvalTriplet{psqt, positional, make_nnue_label(opt, psqt, positional)};
}

}  // namespace nnue_eval_shared

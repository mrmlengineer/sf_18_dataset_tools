/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

#include "uci.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include "movegen.h"
#include "position.h"
#include "score.h"
#include "types.h"

namespace Stockfish {

namespace {

template<typename... Ts>
struct overload: Ts... {
    using Ts::operator()...;
};
template<typename... Ts>
overload(Ts...) -> overload<Ts...>;

constexpr int NormalizeToPawnValue = 361;

}  // namespace

std::string UCIEngine::format_score(const Score& s) {
    constexpr int TB_CP = 20000;
    const auto    format =
      overload{[](Score::Mate mate) -> std::string {
                   auto m = (mate.plies > 0 ? (mate.plies + 1) : mate.plies) / 2;
                   return std::string("mate ") + std::to_string(m);
               },
               [](Score::Tablebase tb) -> std::string {
                   return std::string("cp ")
                        + std::to_string((tb.win ? TB_CP - tb.plies : -TB_CP - tb.plies));
               },
               [](Score::InternalUnits units) -> std::string {
                   return std::string("cp ") + std::to_string(units.value);
               }};

    return s.visit(format);
}

int UCIEngine::to_cp(Value v, const Position&) {
    return int(std::round(100.0 * int(v) / NormalizeToPawnValue));
}

std::string UCIEngine::wdl(Value, const Position&) {
    return "0 1000 0";
}

std::string UCIEngine::square(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string UCIEngine::move(Move m, bool chess960) {
    if (m == Move::none())
        return "(none)";

    if (m == Move::null())
        return "0000";

    Square from = m.from_sq();
    Square to   = m.to_sq();

    if (m.type_of() == CASTLING && !chess960)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    std::string move = square(from) + square(to);

    if (m.type_of() == PROMOTION)
        move += " pnbrqk"[m.promotion_type()];

    return move;
}

std::string UCIEngine::to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return char(std::tolower(c));
    });

    return str;
}

Move UCIEngine::to_move(const Position& pos, std::string str) {
    str = to_lower(std::move(str));

    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == move(m, pos.is_chess960()))
            return m;

    return Move::none();
}

}  // namespace Stockfish

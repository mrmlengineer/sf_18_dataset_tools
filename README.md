# SF18 Dataset Tools

`sf_18_dataset_tools` is a fork of [Stockfish][stockfish-website] based on the
official `sf_18` release/tag. The fork exists to build dataset-generation helper
binaries around Stockfish's board representation, move generation, and
NNUE/static-evaluation code.

This repository is not the official Stockfish project. It keeps the Stockfish
code needed by the helpers, but its supported build workflow is limited
to dataset helper binaries. It does not support building or distributing the
main Stockfish UCI engine executable.

## Helper Binaries

The supported outputs are built under `bin/`:

* `mixed_bucket_fens`, for generating bucketed static-eval rows from move
  streams or FEN-producing workflows.
* `nnue_eval`, for evaluating FENs and exporting NNUE/static-eval components.
* `pgn_to_uci_rows`, for converting PGN input into compact UCI move streams
  with minimum-Elo metadata.

## Building

Build the helper binaries from `src/`:

```sh
cd src
make -j build
```

The default `make`, `make build`, and `make profile-build` targets all build the
dataset helpers only:

* `../bin/mixed_bucket_fens`
* `../bin/nnue_eval`
* `../bin/pgn_to_uci_rows`

Useful Makefile targets:

* `make help`: show supported targets, compilers, and architectures.
* `make dataset-tools`: build all helper binaries.
* `make mixed_bucket_fens`: build only `mixed_bucket_fens`.
* `make nnue_eval`: build only `nnue_eval`.
* `make pgn_to_uci_rows`: build only `pgn_to_uci_rows`.
* `make dataset-tools-clean`: remove helper binaries from `bin/`.

## Repository Layout

Important files and directories:

* `src/`: Stockfish-derived source code plus dataset helper sources.
* `src/tools/dataset/`: dataset-specific helper implementations.
* `bin/`: built helper binaries.
* `Copying.txt`: GPL v3 license text inherited from Stockfish.
* `AUTHORS`: upstream Stockfish authorship file.

## Upstream Stockfish

This fork derives from Stockfish, a free and strong open-source chess engine.
Stockfish is developed by the Stockfish developers and distributed under the GNU
General Public License version 3 or later.

Relevant upstream references:

* [Official Stockfish website][stockfish-website]
* [Official Stockfish repository][stockfish-repo]
* [Official Stockfish source tree][stockfish-src]
* [Stockfish AUTHORS file][stockfish-authors]
* [Stockfish GPL license file][stockfish-license]
* [Stockfish documentation wiki][stockfish-wiki]

## License And Distribution

This fork is distributed under the [GNU General Public License version 3][gpl-v3]
or later, following Stockfish's license.

When distributing this fork, modified source code, or helper binaries built from
it, make sure the distribution includes:

* the GPL license text;
* the corresponding source code needed to build the exact distributed binaries;
* attribution to the Stockfish project and its authors;
* any local modifications required to reproduce the distributed binaries.

See `Copying.txt` for the full license text. This README is only a summary and
does not replace the license.

## Acknowledgements

This project depends on the Stockfish codebase and the work of the Stockfish
developers. Stockfish uses neural networks trained on [data provided by the
Leela Chess Zero project][lc0-data], made available under the
[Open Database License][odbl].

[gpl-v3]: https://www.gnu.org/licenses/gpl-3.0.en.html
[lc0-data]: https://storage.lczero.org/files/training_data
[odbl]: https://opendatacommons.org/licenses/odbl/odbl-10.txt
[stockfish-authors]: https://github.com/official-stockfish/Stockfish/blob/master/AUTHORS
[stockfish-license]: https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt
[stockfish-repo]: https://github.com/official-stockfish/Stockfish
[stockfish-src]: https://github.com/official-stockfish/Stockfish/tree/master/src
[stockfish-website]: https://stockfishchess.org
[stockfish-wiki]: https://github.com/official-stockfish/Stockfish/wiki

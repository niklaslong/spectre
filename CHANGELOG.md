# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0]

### Changed

- Split betweeness and closeness computation into separate functions.

### Added

- Multi-threaded computation of betweenness and closeness metrics.

## [0.6.0]

### Added

- Methods to return the unfiltered or filtered adjacency lists have been added to the public API.
- The `graph!` macro for testing purposes.

### Changed

- Betweenness and closeness metrics have their own methods in the public API.

### Removed

- `Agraph` and `Vertex` types have been removed, betweenness and closeness are
  computed from the adjacency matrix.


## [0.5.2]

### Changed

- Bump `nalgebra` to `v0.32`.

## [0.5.1]

### Removed

- Removed test data from the included files in the package.

## [0.5.0]

### Added

- Added `AGraph` and `Vertex` types.
- Added `Graph::compute_betweenness_and_closeness` based on the `AGraph`.

## [0.4.0]

### Changed

- Update `nalgebra` to `0.31`.
- Change `BTreeMap` to `HashMap` for centrality measurement collections.

## [0.3.0]

### Added

- Create the P2P topology example (`./examples/p2p.rs`).
- Introduce `Graph::insert_subset` and `Graph::update_subset`.

### Changed

- Manually implement `Default` for `Graph`.

### Fixed

- Clear internal cache on `Graph::remove`.

[unreleased]: https://github.com/niklaslong/spectre/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/niklaslong/spectre/compare/v0.5.2...v0.7.0
[0.6.0]: https://github.com/niklaslong/spectre/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/niklaslong/spectre/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/niklaslong/spectre/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/niklaslong/spectre/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/niklaslong/spectre/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/niklaslong/spectre/compare/v0.2.0...v0.3.0


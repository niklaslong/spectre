# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/niklaslong/spectre/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/niklaslong/spectre/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/niklaslong/spectre/compare/v0.2.0...v0.3.0


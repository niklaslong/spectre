//! Spectre is a small toolkit for analysing p2p network graphs, though it can also apply more generally
//! to undirected graphs.
//!
//! # Basic usage
//!
//! The library is centered around the [`Graph`](graph::Graph) structure which can be constructed
//! from one or more [`Edge`](edge::Edge) instances. Once constructed, various measurements and
//! matrix representations of the graph can be computed.
//!
//! ```rust
//! use std::net::SocketAddr;
//!
//! use spectre::edge::Edge;
//! use spectre::graph::Graph;
//!
//! // Construct the graph instance.
//! let mut graph = Graph::new();
//!
//! // Create some addresses to be part of a network topology.
//! let addrs: Vec<SocketAddr> = (0..3)
//!     .map(|i| format!("127.0.0.1:{i}").parse().unwrap())
//!     .collect();
//! let (a, b, c) = (addrs[0], addrs[1], addrs[2]);
//!
//! // Insert some edges, note the IDs can be any type that is `Copy + Eq + Hash + Ord`.
//! graph.insert(Edge::new(a, b));
//! graph.insert(Edge::new(a, c));
//!
//! // Compute some metrics on that state of the graph.
//! let density = graph.density();
//! let degree_centrality_delta = graph.degree_centrality_delta();
//!
//! // Matrices can be pretty printed...
//! println!("{}", graph.laplacian_matrix());
//! // ...outputs:
//! //  ┌          ┐
//! //  │  2 -1 -1 │
//! //  │ -1  1  0 │
//! //  │ -1  0  1 │
//! //  └          ┘
//! ```

mod betweenness;
mod closeness;
pub mod edge;
pub mod graph;

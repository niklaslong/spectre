use std::{
    collections::{BTreeMap, HashSet},
    hash::Hash,
};

use nalgebra::DMatrix;

use crate::edge::Edge;

/// An undirected graph, made up of edges.
#[derive(Clone, Debug, Default)]
pub struct Graph<T> {
    /// The edges in the graph.
    edges: HashSet<Edge<T>>,
    /// A mapping of vertices to their indices to be used when constructing the various matrices
    /// representing the graph.
    ///
    /// The use of a `BTreeMap` means we need the `Ord` bound on `T`. The sorted collection allows
    /// us to maintain some form of order between computations, which can be useful for debugging.
    index: Option<BTreeMap<T, usize>>,
}

impl<T> Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord,
{
    /// Inserts an edge into the graph.
    pub fn insert(&mut self, edge: Edge<T>) -> bool {
        let is_inserted = self.edges.insert(edge);

        // Delete the index if the edge was successfully inserted because we can't reliably update
        // it from the new connection alone.
        if is_inserted && self.index.is_some() {
            self.index = None
        }

        is_inserted
    }

    /// Checks if the graph contains an edge.
    pub fn contains(&self, edge: &Edge<T>) -> bool {
        self.edges.contains(edge)
    }

    /// Returns the vertex count of the graph.
    pub fn vertex_count(&self) -> usize {
        self.vertices_from_edges().len()
    }

    /// Returns the edge count of the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Computes the density of the graph, the ratio of edges with respect to the maximum possible
    /// edges.
    pub fn density(&self) -> f64 {
        let vc = self.vertex_count() as f64;
        let ec = self.edge_count() as f64;

        // Calculate the total number of possible edges given a vertex count.
        let pec = vc * (vc - 1.0) / 2.0;
        // Actual edges divided by the possible edges gives the density.
        ec / pec
    }

    /// Generates the adjacency matrix for this graph.
    pub fn adjacency_matrix(&mut self) -> DMatrix<f64> {
        self.generate_index();

        // Safety: the previous call guarantees the index has been generated and stored.
        let n = self.index.as_ref().unwrap().len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);

        // Compute the adjacency matrix. As our we're assuming the graph is undirected, the adjacency matrix is
        // symmetric.
        for edge in &self.edges {
            // Safety: get the indices for each edge in the graph, these must be present as the
            // index was generated from this set of edges.
            let i = self.index.as_ref().unwrap().get(edge.source()).unwrap();
            let j = self.index.as_ref().unwrap().get(edge.target()).unwrap();

            // Since edges are guaranteed to be unique, both the upper and lower triangles must be
            // writted (as the graph is unidrected) for each edge.
            matrix[(*i, *j)] = 1.0;
            matrix[(*j, *i)] = 1.0;
        }

        matrix
    }

    // Private API

    /// Returns the set of unique vertices contained within the set of edges.
    fn vertices_from_edges(&self) -> HashSet<T> {
        let mut vertices: HashSet<T> = HashSet::new();
        for edge in self.edges.iter() {
            // Using a hashset guarantees uniqueness.
            vertices.insert(*edge.source());
            vertices.insert(*edge.target());
        }

        vertices
    }

    /// Constructs and stores an index of vertices for this set of edges.
    fn generate_index(&mut self) {
        let vertices = self.vertices_from_edges();

        let index: BTreeMap<T, usize> = vertices
            .iter()
            .enumerate()
            .map(|(i, &vertex)| (vertex, i))
            .collect();

        self.index = Some(index);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    #[test]
    fn insert() {
        let mut graph = Graph::default();
        let edge = Edge::new("a", "b");

        assert!(graph.insert(edge.clone()));
        assert!(!graph.insert(edge));
    }

    #[test]
    fn contains() {
        let mut graph = Graph::default();
        let edge = Edge::new("a", "b");

        graph.insert(edge.clone());

        assert!(graph.contains(&edge));
        assert!(!graph.contains(&Edge::new("b", "c")));
    }

    #[test]
    fn vertex_count() {
        let mut graph = Graph::default();
        assert_eq!(graph.vertex_count(), 0);

        // Verify two new vertices get added when they don't yet exist in the graph.
        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.vertex_count(), 2);

        // Verify only one new vertex is added when one of them already exists in the graph.
        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.vertex_count(), 3);
    }

    #[test]
    fn edge_count() {
        let mut graph = Graph::default();
        assert_eq!(graph.edge_count(), 0);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn density() {
        let mut graph = Graph::default();
        assert!(graph.density().is_nan());

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.density(), 1.0);

        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.density(), 2.0 / 3.0);
    }

    #[test]
    fn adjacency_matrix() {
        let mut graph = Graph::default();
        assert_eq!(graph.adjacency_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.adjacency_matrix(),
            dmatrix![0.0, 1.0;
                     1.0, 0.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some())
    }
}

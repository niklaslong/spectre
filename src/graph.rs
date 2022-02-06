use std::{collections::HashSet, hash::Hash};

use crate::edge::Edge;

/// An undirected graph, made up of edges.
#[derive(Clone, Debug, Default)]
pub struct Graph<T>(HashSet<Edge<T>>);

impl<T> Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash,
{
    /// Inserts an edge into the graph.
    pub fn insert(&mut self, edge: Edge<T>) -> bool {
        self.0.insert(edge)
    }

    /// Checks if the graph contains an edge.
    pub fn contains(&self, edge: &Edge<T>) -> bool {
        self.0.contains(edge)
    }

    /// Returns the vertex count of the graph.
    pub fn vertex_count(&self) -> usize {
        self.vertices_from_edges().len()
    }

    /// Returns the edge count of the graph.
    pub fn edge_count(&self) -> usize {
        self.0.len()
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

    // Private API

    fn vertices_from_edges(&self) -> HashSet<T> {
        let mut vertices: HashSet<T> = HashSet::new();
        for edge in self.0.iter() {
            // Using a hashset guarantees uniqueness.
            vertices.insert(*edge.source());
            vertices.insert(*edge.target());
        }

        vertices
    }
}

#[cfg(test)]
mod tests {
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
}

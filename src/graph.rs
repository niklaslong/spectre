use std::{
    collections::{BTreeMap, HashSet},
    hash::Hash,
    ops::Sub,
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
    /// Cache the degree matrix when possible.
    degree_matrix: Option<DMatrix<f64>>,
    /// Cache the adjacency matrix when possible.
    adjacency_matrix: Option<DMatrix<f64>>,
    /// Cache the laplacian matrix when possible.
    laplacian_matrix: Option<DMatrix<f64>>,
}

impl<T> Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord,
{
    /// Inserts an edge into the graph.
    pub fn insert(&mut self, edge: Edge<T>) -> bool {
        let is_inserted = self.edges.insert(edge);

        // Delete the cached objects if the edge was successfully inserted because we can't
        // reliably update them from the new connection alone.
        if is_inserted && self.index.is_some() {
            self.clear_cache()
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

    /// Constructs the adjacency matrix for this graph.
    pub fn adjacency_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.adjacency_matrix.clone() {
            return matrix;
        }

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

        // Cache the matrix.
        self.adjacency_matrix = Some(matrix.clone());

        matrix
    }

    /// Constructs the degree matrix for this graph.
    pub fn degree_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.degree_matrix.clone() {
            return matrix;
        }

        let adjacency_matrix = self.adjacency_matrix();

        // Safety: the previous call guarantees the index has been generated and stored.
        let n = self.index.as_ref().unwrap().len();
        let mut matrix = DMatrix::<f64>::zeros(n, n);

        for (i, row) in adjacency_matrix.row_iter().enumerate() {
            // Set the diagonal to be the sum of edges in that row. The index isn't necessary
            // here since the rows are visited in order and the adjacency matrix is ordered after the
            // index.
            matrix[(i, i)] = row.sum()
        }

        // Cache the matrix.
        self.degree_matrix = Some(matrix.clone());

        matrix
    }

    /// Constructs the laplacian matrix for this graph.
    pub fn laplacian_matrix(&mut self) -> DMatrix<f64> {
        // Check the cache.
        if let Some(matrix) = self.laplacian_matrix.clone() {
            return matrix;
        }

        let degree_matrix = self.degree_matrix();
        let adjacency_matrix = self.adjacency_matrix();

        let matrix = degree_matrix.sub(&adjacency_matrix);

        // Cache the matrix.
        self.laplacian_matrix = Some(matrix.clone());

        matrix
    }

    //
    // Private
    //

    fn clear_cache(&mut self) {
        self.index = None;
        self.degree_matrix = None;
        self.adjacency_matrix = None;
        self.laplacian_matrix = None;
    }

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
    ///
    /// The index will be sorted by `T`'s implementation of `Ord`.
    fn generate_index(&mut self) {
        // It should be impossible to call this function if the cache is not empty.
        debug_assert!(self.index.is_none());

        let mut vertices: Vec<T> = self.vertices_from_edges().into_iter().collect();
        vertices.sort();

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

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.adjacency_matrix(),
            dmatrix![0.0, 1.0, 1.0;
                     1.0, 0.0, 0.0;
                     1.0, 0.0, 0.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    #[test]
    fn degree_matrix() {
        let mut graph = Graph::default();
        assert_eq!(graph.degree_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.degree_matrix(),
            dmatrix![1.0, 0.0;
                     0.0, 1.0]
        );

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.degree_matrix(),
            dmatrix![2.0, 0.0, 0.0;
                     0.0, 1.0, 0.0;
                     0.0, 0.0, 1.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    #[test]
    fn laplacian_matrix() {
        let mut graph = Graph::default();
        assert_eq!(graph.degree_matrix(), dmatrix![]);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(
            graph.laplacian_matrix(),
            dmatrix![1.0, -1.0;
                     -1.0, 1.0]
        );

        graph.insert(Edge::new("a", "c"));
        assert_eq!(
            graph.laplacian_matrix(),
            dmatrix![2.0, -1.0, -1.0;
                     -1.0, 1.0, 0.0;
                     -1.0, 0.0, 1.0]
        );

        // Sanity check the index gets stored.
        assert!(graph.index.is_some());
    }

    //
    // Private
    //

    #[test]
    fn clear_cache_on_insert() {
        let mut graph = Graph::default();
        graph.insert(Edge::new("a", "b"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with an insert.
        graph.insert(Edge::new("a", "c"));

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn vertices_from_edges() {
        let mut graph = Graph::default();
        assert!(graph.vertices_from_edges().is_empty());

        let (a, b) = ("a", "b");
        graph.insert(Edge::new(a, b));

        let vertices = graph.vertices_from_edges();
        assert!(vertices.contains(a));
        assert!(vertices.contains(b));

        // Sanity check the length.
        assert_eq!(vertices.len(), 2);
    }

    #[test]
    fn generate_index() {
        let mut graph = Graph::default();

        // Check for an empty graph.
        graph.generate_index();
        assert!(graph.index.is_some());
        assert!(graph.index.as_ref().unwrap().is_empty());

        let (a, b) = ("a", "b");
        graph.insert(Edge::new(a, b));
        graph.generate_index();

        assert!(graph.index.is_some());

        assert_eq!(
            graph.index.as_ref().unwrap().get_key_value(a),
            Some((&a, &(0 as usize)))
        );

        assert_eq!(
            graph.index.as_ref().unwrap().get_key_value(b),
            Some((&b, &(1 as usize)))
        );

        assert_eq!(graph.index.as_ref().unwrap().len(), 2);
    }
}

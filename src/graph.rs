//! A module for working with graphs.

use std::{
    collections::{BTreeMap, HashSet},
    hash::Hash,
    ops::Sub,
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

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
    /// Creates an empty graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::graph::Graph;
    ///
    /// let graph: Graph<&str> = Graph::new();
    /// ```
    pub fn new() -> Self
    where
        T: Default,
    {
        Default::default()
    }

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

    /// Removes an edge from the set and returns whether it was present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    ///
    /// assert_eq!(graph.remove(&Edge::new("a", "b")), true);
    /// assert_eq!(graph.remove(&Edge::new("a", "c")), false);
    /// ```
    pub fn remove(&mut self, edge: &Edge<T>) -> bool {
        self.edges.remove(edge)
    }

    /// Checks if the graph contains an edge.
    pub fn contains(&self, edge: &Edge<T>) -> bool {
        self.edges.contains(edge)
    }

    /// Returns the vertex count of the graph.
    ///
    /// This call constructs the collection of vertices from the collection of edges. This is
    /// because the vertex set can't accurately be updated on the basis of the addition or the
    /// removal of an edge alone.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    ///
    /// assert_eq!(graph.vertex_count(), 2);
    /// ```
    pub fn vertex_count(&self) -> usize {
        self.vertices_from_edges().len()
    }

    /// Returns the edge count of the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Computes the density of the graph, the ratio of edges with respect to the maximum possible
    /// edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    ///
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(graph.density(), 1.0);
    ///
    /// graph.insert(Edge::new("a", "c"));
    /// assert_eq!(graph.density(), 2.0 / 3.0);
    /// ```
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

    /// Returns the difference between the highest and lowest degree centrality in the network.
    ///
    /// Returns an `f64`, though the value should be a natural number.
    pub fn degree_centrality_delta(&mut self) -> f64 {
        let degree_matrix = self.degree_matrix();

        let max = degree_matrix.diagonal().max();
        let min = degree_matrix.diagonal().min();

        max - min
    }

    /// Returns a mapping of the edges to their degree centrality (number of connections) in the graph.
    pub fn degree_centrality(&mut self) -> BTreeMap<T, u32> {
        let degree_matrix = self.degree_matrix();

        // Safety: the previous call guarantees the index has been generated and stored.
        self.index
            .as_ref()
            .unwrap()
            .keys()
            .zip(degree_matrix.diagonal().iter())
            .map(|(addr, dc)| (*addr, *dc as u32))
            .collect()
    }

    /// Returns a mapping of the edges to their eigenvalue centrality (the relative importance of the
    /// edge) in the graph.
    pub fn eigenvalue_centrality(&mut self) -> BTreeMap<T, f64> {
        let adjacency_matrix = self.adjacency_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if adjacency_matrix.is_empty() {
            return BTreeMap::new();
        }

        // Compute the eigenvectors and corresponding eigenvalues and sort in descending order.
        let ascending = false;
        let eigenvalue_vector_pairs = sorted_eigenvalue_vector_pairs(adjacency_matrix, ascending);
        let (_highest_eigenvalue, highest_eigenvector) = &eigenvalue_vector_pairs[0];

        // The eigenvector is a relative score of node importance (normalised by the norm), to obtain an absolute score for each
        // node, we normalise so that the sum of the components are equal to 1.
        let sum = highest_eigenvector.sum() / self.index.as_ref().unwrap().len() as f64;
        let normalised = highest_eigenvector.unscale(sum);

        // Map addresses to their eigenvalue centrality.
        self.index
            .as_ref()
            .unwrap()
            .keys()
            .zip(normalised.column(0).iter())
            .map(|(addr, ec)| (*addr, *ec))
            .collect()
    }

    /// Returns the algebraic connectivity (Fiedler eigenvalue) of the graph and a mapping of the
    /// edges to their Fiedler value (their associated component in the Fiedler eigenvector).
    pub fn fiedler(&mut self) -> (f64, BTreeMap<T, f64>) {
        let laplacian_matrix = self.laplacian_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if laplacian_matrix.is_empty() {
            return (0.0, BTreeMap::new());
        }

        // Compute the eigenvectors and corresponding eigenvalues and sort in ascending order.
        let ascending = true;
        let pairs = sorted_eigenvalue_vector_pairs(laplacian_matrix, ascending);

        // Second-smallest eigenvalue of the Laplacian is the Fiedler value (algebraic connectivity), the associated
        // eigenvector is the Fiedler vector.
        let (algebraic_connectivity, fiedler_vector) = &pairs[1];

        // Map addresses to their Fiedler values.
        let fiedler_values_indexed = self
            .index
            .as_ref()
            .unwrap()
            .keys()
            .zip(fiedler_vector.column(0).iter())
            .map(|(addr, fiedler_value)| (*addr, *fiedler_value))
            .collect();

        (*algebraic_connectivity, fiedler_values_indexed)
    }

    //
    // Private
    //

    /// Clears the computed state.
    ///
    /// This should be called every time the set of edges is mutated since the cached state won't
    /// correspond to the new graph.
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

//
// Helpers
//

/// Computes the eigenvalues and corresponding eigenvalues from the supplied symmetric matrix.
fn sorted_eigenvalue_vector_pairs(
    matrix: DMatrix<f64>,
    ascending: bool,
) -> Vec<(f64, DVector<f64>)> {
    // Early return if the matrix is empty, the rest of the computation requires a matrix with
    // at least a dim of 1x1.
    if matrix.is_empty() {
        return vec![];
    }

    // Compute eigenvalues and eigenvectors.
    let eigen = SymmetricEigen::new(matrix);

    // Map eigenvalues to their eigenvectors.
    let mut pairs: Vec<(f64, DVector<f64>)> = eigen
        .eigenvalues
        .iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(value, vector)| (*value, vector.clone_owned()))
        .collect();

    // Sort eigenvalue-vector pairs in descending order.
    pairs.sort_unstable_by(|(a, _), (b, _)| {
        if ascending {
            a.partial_cmp(b).unwrap()
        } else {
            b.partial_cmp(a).unwrap()
        }
    });

    pairs
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    #[test]
    fn new() {
        let _: Graph<()> = Graph::new();
    }

    #[test]
    fn insert() {
        let mut graph = Graph::new();
        let edge = Edge::new("a", "b");

        assert!(graph.insert(edge.clone()));
        assert!(!graph.insert(edge));
    }

    #[test]
    fn remove() {
        let edge = Edge::new("a", "b");
        let uninserted_edge = Edge::new("a", "c");

        let mut graph = Graph::new();
        graph.insert(edge.clone());

        assert!(graph.remove(&edge));
        assert!(!graph.remove(&uninserted_edge));
    }

    #[test]
    fn contains() {
        let mut graph = Graph::new();
        let edge = Edge::new("a", "b");

        graph.insert(edge.clone());

        assert!(graph.contains(&edge));
        assert!(!graph.contains(&Edge::new("b", "c")));
    }

    #[test]
    fn vertex_count() {
        let mut graph = Graph::new();
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
        let mut graph = Graph::new();
        assert_eq!(graph.edge_count(), 0);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn density() {
        let mut graph = Graph::new();
        assert!(graph.density().is_nan());

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.density(), 1.0);

        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.density(), 2.0 / 3.0);
    }

    #[test]
    fn adjacency_matrix() {
        let mut graph = Graph::new();
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
        let mut graph = Graph::new();
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
        let mut graph = Graph::new();
        assert_eq!(graph.laplacian_matrix(), dmatrix![]);

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

    #[test]
    fn degree_centrality_delta() {
        let mut graph = Graph::new();
        assert_eq!(graph.degree_centrality_delta(), 0.0);

        graph.insert(Edge::new("a", "b"));
        assert_eq!(graph.degree_centrality_delta(), 0.0);

        graph.insert(Edge::new("a", "c"));
        assert_eq!(graph.degree_centrality_delta(), 1.0);
    }

    #[test]
    fn degree_centrality() {
        let mut graph = Graph::new();
        assert!(graph.degree_centrality().is_empty());

        // One connection, centrality measures for each vertex should be 1.
        let (a, b, c) = ("a", "b", "c");
        graph.insert(Edge::new(a, b));
        let degree_centrality = graph.degree_centrality();

        assert_eq!(degree_centrality.get_key_value(a), Some((&a, &1)));
        assert_eq!(degree_centrality.get_key_value(b), Some((&b, &1)));

        // Sanity check the length.
        assert_eq!(degree_centrality.len(), 2);

        // Two connections, degree centrality for A should increase.
        graph.insert(Edge::new(a, c));
        let degree_centrality = graph.degree_centrality();

        assert_eq!(degree_centrality.get_key_value(a), Some((&a, &2)));
        assert_eq!(degree_centrality.get_key_value(b), Some((&b, &1)));
        assert_eq!(degree_centrality.get_key_value(c), Some((&c, &1)));

        // Sanity check the length.
        assert_eq!(degree_centrality.len(), 3);
    }

    #[test]
    fn eigenvalue_centrality() {
        let mut graph = Graph::new();
        assert!(graph.eigenvalue_centrality().is_empty());

        // One connection, centrality measures for each vertex should be 1.
        let (a, b, c) = ("a", "b", "c");
        graph.insert(Edge::new(a, b));
        let eigenvalue_centrality = graph.eigenvalue_centrality();

        assert_eq!(eigenvalue_centrality.get_key_value(a), Some((&a, &1.0)));
        assert_eq!(eigenvalue_centrality.get_key_value(b), Some((&b, &1.0)));

        // Sanity check the length.
        assert_eq!(eigenvalue_centrality.len(), 2);

        // Two connections, degree centrality for A should increase.
        graph.insert(Edge::new(a, c));
        let eigenvalue_centrality = graph.eigenvalue_centrality();

        assert_eq!(
            eigenvalue_centrality.get_key_value(a),
            Some((&a, &1.2426406871192854))
        );
        assert_eq!(
            eigenvalue_centrality.get_key_value(b),
            Some((&b, &0.8786796564403571))
        );
        assert_eq!(
            eigenvalue_centrality.get_key_value(c),
            Some((&c, &0.8786796564403576))
        );

        // Sanity check the length.
        assert_eq!(eigenvalue_centrality.len(), 3);
    }

    #[test]
    fn fiedler() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        // Disconnected graph.
        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(c, d));

        // Algebraic connectivity should be 0.
        let (algebraic_connectivity, fiedler_values_indexed) = graph.fiedler();
        assert_eq!(algebraic_connectivity, 0.0);
        assert_eq!(fiedler_values_indexed.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(fiedler_values_indexed.get_key_value(b), Some((&b, &0.0)));
        assert_eq!(
            fiedler_values_indexed.get_key_value(c),
            Some((&c, &-0.7071067811865475))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(d),
            Some((&d, &-0.7071067811865475))
        );

        // Connect the graph.
        graph.insert(Edge::new(b, c));

        let (algebraic_connectivity, fiedler_values_indexed) = graph.fiedler();
        assert_eq!(algebraic_connectivity, 0.5857864376269044);
        assert_eq!(
            fiedler_values_indexed.get_key_value(a),
            Some((&a, &0.6532814824381882))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(b),
            Some((&b, &0.27059805007309845))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(c),
            Some((&c, &-0.2705980500730985))
        );
        assert_eq!(
            fiedler_values_indexed.get_key_value(d),
            Some((&d, &-0.6532814824381881))
        );
    }

    //
    // Private
    //

    #[test]
    fn clear_cache_on_insert() {
        let mut graph = Graph::new();
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
        let mut graph = Graph::new();
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
        let mut graph = Graph::new();

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

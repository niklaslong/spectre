//! A module for working with graphs.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
    ops::Sub,
};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{compute::compute_betweenness, edge::Edge};

// for performance reasons, we keep the
// index size as small as possible
pub type GraphIndex = u16;
const MAX_INDICES_NODES: usize = u16::MAX as usize;

/// An undirected graph, made up of edges.
#[derive(Clone, Debug)]
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
    /// Cache the betweenness count when possible.
    betweenness_count: Option<Vec<f64>>,
    /// Cache the path lengths when possible.
    total_path_length: Option<Vec<u32>>,
}

impl<T> Default for Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord,
{
    fn default() -> Self {
        Self::new()
    }
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
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
            index: None,
            degree_matrix: None,
            adjacency_matrix: None,
            laplacian_matrix: None,
            betweenness_count: None,
            total_path_length: None,
        }
    }

    pub fn edges(&mut self) -> &HashSet<Edge<T>> {
        &self.edges
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

    /// Inserts a subset of `(hub, leaf)` edges into the graph.
    pub fn insert_subset(&mut self, hub: T, leaves: &[T]) {
        for leaf in leaves {
            self.insert(Edge::new(hub, *leaf));
        }
    }

    /// Inserts a subset of `(hub, leaf)` edges into the graph and removes any existing edges that
    /// contain the hub but aren't included in the new set.
    pub fn update_subset(&mut self, hub: T, leaves: &[T]) {
        let new_edges: HashSet<Edge<T>> = leaves.iter().map(|leaf| Edge::new(hub, *leaf)).collect();

        // Remove hub-containing edges that aren't included in the new set.
        let original_len = self.edge_count();
        self.edges
            .retain(|edge| new_edges.contains(edge) || !edge.contains(&hub));

        // Make sure to clear the cache after removals as there may be no inserts.
        // TODO: make more efficient.
        if self.edge_count() != original_len {
            self.clear_cache()
        }

        for edge in new_edges {
            self.insert(edge);
        }
    }

    /// Removes an edge from the set and returns whether it was present in the set.
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
    /// assert_eq!(graph.remove(&Edge::new("a", "b")), true);
    /// assert_eq!(graph.remove(&Edge::new("a", "c")), false);
    /// ```
    pub fn remove(&mut self, edge: &Edge<T>) -> bool {
        let is_removed = self.edges.remove(edge);

        // Delete the cached objects if the edge was successfully removed because we can't reliably
        // update them from the new connection alone.
        if is_removed && self.index.is_some() {
            self.clear_cache()
        }

        is_removed
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
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.adjacency_matrix(),
    ///     dmatrix![0.0, 1.0;
    ///              1.0, 0.0]
    /// );
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.degree_matrix(),
    ///     dmatrix![1.0, 0.0;
    ///              0.0, 1.0]
    /// );
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::dmatrix;
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// assert_eq!(
    ///     graph.laplacian_matrix(),
    ///     dmatrix![1.0, -1.0;
    ///              -1.0, 1.0]
    /// );
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use spectre::edge::Edge;
    /// use spectre::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.insert(Edge::new("a", "b"));
    /// graph.insert(Edge::new("a", "c"));
    ///
    /// assert_eq!(graph.degree_centrality_delta(), 1.0);
    /// ```
    pub fn degree_centrality_delta(&mut self) -> f64 {
        let degree_matrix = self.degree_matrix();

        let max = degree_matrix.diagonal().max();
        let min = degree_matrix.diagonal().min();

        max - min
    }

    /// Returns a mapping of vertices to their degree centrality (number of connections) in the graph.
    pub fn degree_centrality(&mut self) -> HashMap<T, u32> {
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

    /// Returns a mapping of vertices to their eigenvalue centrality (the relative importance of
    /// the vertex) in the graph.
    pub fn eigenvalue_centrality(&mut self) -> HashMap<T, f64> {
        let adjacency_matrix = self.adjacency_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if adjacency_matrix.is_empty() {
            return HashMap::new();
        }

        // Compute the eigenvectors and corresponding eigenvalues and sort in descending order.
        let ascending = false;
        let eigenvalue_vector_pairs = sorted_eigenvalue_vector_pairs(adjacency_matrix, ascending);
        let (_highest_eigenvalue, highest_eigenvector) = &eigenvalue_vector_pairs[0];

        // The eigenvector is a relative score of vertex importance (normalised by the norm), to obtain an absolute score for each
        // vertex, we normalise so that the sum of the components are equal to 1.
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
    /// vertices to their Fiedler value (their associated component in the Fiedler eigenvector).
    pub fn fiedler(&mut self) -> (f64, HashMap<T, f64>) {
        let laplacian_matrix = self.laplacian_matrix();

        // Early return if the matrix is empty, the rest of the computation requires a matrix with
        // at least a dim of 1x1.
        if laplacian_matrix.is_empty() {
            return (0.0, HashMap::new());
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
        self.betweenness_count = None;
        self.total_path_length = None;
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

    /// This method returns a set connection indices for each node.
    /// It a compact way to view the adjacency matrix, and therefore, is
    /// used for the computation of betweenness and closeness centralities
    pub fn get_adjacency_indices(&mut self) -> Vec<Vec<GraphIndex>> {
        let mut indices: Vec<Vec<GraphIndex>> = Vec::new();
        let adjacency_matrix = self.adjacency_matrix();

        assert!(
            adjacency_matrix.nrows() <= MAX_INDICES_NODES,
            "The number of nodes in the graph {} exceeds the maximum number allowed {}",
            indices.len(),
            MAX_INDICES_NODES
        );

        for m in 0..adjacency_matrix.nrows() {
            let neighbors: Vec<GraphIndex> = adjacency_matrix
                .row(m)
                .iter()
                .enumerate()
                .filter(|(_n, &val)| val == 1.0)
                .map(|(n, _)| n as GraphIndex)
                .collect();
            indices.push(neighbors);
        }
        indices
    }

    /// This method also outputs an array of index vectors, although it is created differently.
    /// It is currently used if filtering of nodes is required.
    pub fn get_filtered_adjacency_indices(&self, nodes_to_keep: &Vec<T>) -> Vec<Vec<usize>> {
        let num_nodes = nodes_to_keep.len();
        let mut indices = Vec::with_capacity(num_nodes);
        let mut node_map = HashMap::with_capacity(num_nodes);
        for (n, node) in nodes_to_keep.iter().enumerate().take(num_nodes) {
            // make initial capacity 10% of total
            indices.push(Vec::with_capacity(num_nodes / 10));
            node_map.insert(node, n);
        }

        // For each edge, check if the source and target nodes
        // are in our node HashMap.  If we've obtained both
        // indices, insert into the corresponding connection list
        for edge in self.edges.iter() {
            if let Some(source_index) = node_map.get(edge.source()) {
                if let Some(target_index) = node_map.get(edge.target()) {
                    indices[*source_index].push(*target_index);
                    indices[*target_index].push(*source_index);
                }
            }
        }

        for node in indices.iter_mut() {
            node.shrink_to_fit();
        }

        indices
    }

    /// This method computes the closeness and betweenness for a given Graph.
    ///
    /// Closeness: for each node, find all shortest paths to all other nodes.
    /// Accumulate all path lengths, accumulate number of paths, and then compute
    /// average path length.
    ///
    /// Betweenness: When a shortest path is found, for all nodes
    /// in-between (i.e., not an end point), increment their betweenness value.
    /// Normalize the counts by dividing by the number of shortest paths found
    ///
    fn betweenness_and_closeness_centrality(&mut self, num_threads: usize) {
        if self.betweenness_count.is_some() {
            return;
        }

        let (betweenness_count, total_path_length) =
            compute_betweenness(self.get_adjacency_indices(), num_threads);

        self.betweenness_count = Some(betweenness_count);
        self.total_path_length = Some(total_path_length);
    }

    /// This method returns the betweenness for a given Graph.
    ///
    /// Betweenness: When a shortest path is found, for all nodes
    /// in-between (i.e., not an end point), increment their betweenness value.
    /// Normalize the counts by dividing by the number of shortest paths found
    ///
    pub fn betweenness_centrality(&mut self, num_threads: usize) -> HashMap<T, f64> {
        self.betweenness_and_closeness_centrality(num_threads);

        let betweenness_count = self.betweenness_count.as_ref().unwrap();

        let mut centralities = HashMap::new();
        for (node, i) in self.index.as_ref().unwrap() {
            let value = betweenness_count[*i];
            centralities.insert(*node, value);
        }
        centralities
    }

    /// This method returns the closeness for a given Graph.
    ///
    /// Closeness: for each node, find all shortest paths to all other nodes.
    /// Accumulate all path lengths, accumulate number of paths, and then compute
    /// average path length.
    pub fn closeness_centrality(&mut self, num_threads: usize) -> HashMap<T, f64> {
        self.betweenness_and_closeness_centrality(num_threads);

        let total_path_length = self.total_path_length.as_ref().unwrap();

        let mut centralities = HashMap::new();
        let divisor: f64 = total_path_length.len() as f64 - 1.0;
        for (n, node) in self.index.as_ref().unwrap().keys().enumerate() {
            let value = total_path_length[n] as f64 / divisor;
            centralities.insert(*node, value);
        }

        centralities
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
    use std::fs;

    use nalgebra::dmatrix;
    use serde::Deserialize;

    use super::*;

    #[derive(Default, Clone, Deserialize, Debug)]
    pub struct Sample {
        pub node_ips: Vec<String>,
        pub indices: Vec<Vec<usize>>,
    }

    // Creates a graph from a list of paths (that can overlap, the graph handles deduplication).
    macro_rules! graph {
        ($($path:expr),*) => {{
            let mut graph = Graph::new();

            $(
                let mut iter = $path.into_iter().peekable();
                while let (Some(a), Some(b)) = (iter.next(), iter.peek()) {
                    graph.insert(Edge::new(a, b));
                }

            )*

            graph
        }}
    }

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
    fn insert_subset() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(a, c));

        let edges = vec![b, d];
        graph.insert_subset(a, &edges);

        assert!(graph.contains(&Edge::new(a, b)));
        assert!(graph.contains(&Edge::new(a, c)));
        assert!(graph.contains(&Edge::new(a, d)));

        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn update_subset() {
        let mut graph = Graph::new();

        let (a, b, c, d) = ("a", "b", "c", "d");

        graph.insert(Edge::new(a, b));
        graph.insert(Edge::new(a, c));
        graph.insert(Edge::new(b, c));

        let edges = vec![b, d];
        graph.update_subset(a, &edges);

        assert!(graph.contains(&Edge::new(a, b)));
        assert!(!graph.contains(&Edge::new(a, c)));
        assert!(graph.contains(&Edge::new(b, c)));
        assert!(graph.contains(&Edge::new(a, d)));

        assert_eq!(graph.edge_count(), 3);
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
    fn clear_cache_on_subset_insert() {
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

        // Update the graph with a subset insert.
        graph.insert_subset("a", &["b", "d"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_subset_update() {
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

        // Update the graph with a subset update.
        graph.update_subset("a", &["b", "d"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_subset_update_w_only_removals() {
        let mut graph = Graph::new();
        graph.insert(Edge::new("a", "b"));
        graph.insert(Edge::new("a", "c"));

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with a subset update.
        graph.update_subset("a", &["b"]);

        // Check the cache has been cleared.
        assert!(graph.index.is_none());
        assert!(graph.adjacency_matrix.is_none());
        assert!(graph.degree_matrix.is_none());
        assert!(graph.laplacian_matrix.is_none());
    }

    #[test]
    fn clear_cache_on_remove() {
        let edge = Edge::new("a", "b");
        let mut graph = Graph::new();
        graph.insert(edge.clone());

        // The laplacian requires the computation of the index, the degree matrix and the adjacency
        // matrix.
        graph.laplacian_matrix();

        // Check the objects have been cached.
        assert!(graph.index.is_some());
        assert!(graph.adjacency_matrix.is_some());
        assert!(graph.degree_matrix.is_some());
        assert!(graph.laplacian_matrix.is_some());

        // Update the graph with remove.
        graph.remove(&edge);

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
            Some((&a, &0))
        );

        assert_eq!(
            graph.index.as_ref().unwrap().get_key_value(b),
            Some((&b, &1))
        );

        assert_eq!(graph.index.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_graph() {
        let mut graph: Graph<usize> = Graph::new();
        // this graph reproduces the image at:
        // https://www.youtube.com/watch?v=ptqt2zr9ZRE
        graph.insert(Edge::new(0, 1));
        graph.insert(Edge::new(1, 3));
        graph.insert(Edge::new(3, 4));
        graph.insert(Edge::new(4, 2));
        graph.insert(Edge::new(2, 0));
        graph.insert(Edge::new(4, 5));
        graph.insert(Edge::new(5, 3));

        let between_map = graph.betweenness_centrality(1);
        let close_map = graph.closeness_centrality(1);
        let mut betweenness: [f64; 6] = [0.0; 6];
        let mut closeness: [f64; 6] = [0.0; 6];
        for i in 0..6 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [9, 8, 8, 7, 7, 9];
        let mut expected_closeness: [f64; 6] = [0.0; 6];
        let expected_betweenness: [f64; 6] = [1.0, 1.5, 1.5, 2.5, 2.5, 0.0];
        for i in 0..6 {
            expected_closeness[i] = total_path_length[i] as f64 / 5.0;
        }

        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn randomish_graph() {
        let mut graph: Graph<usize> = Graph::new();
        graph.insert(Edge::new(0, 3));
        graph.insert(Edge::new(0, 5));
        graph.insert(Edge::new(5, 1));
        graph.insert(Edge::new(1, 2));
        graph.insert(Edge::new(2, 4));
        graph.insert(Edge::new(2, 6));
        graph.insert(Edge::new(1, 3));

        let between_map = graph.betweenness_centrality(1);
        let close_map = graph.closeness_centrality(1);
        let mut betweenness: [f64; 7] = [0.0; 7];
        let mut closeness: [f64; 7] = [0.0; 7];
        for i in 0..7 {
            betweenness[i] = *between_map.get(&i).unwrap();
            closeness[i] = *close_map.get(&i).unwrap();
        }

        let total_path_length = [15, 9, 10, 12, 15, 12, 15];
        let mut expected_closeness: [f64; 7] = [0.0; 7];
        let expected_betweenness: [f64; 7] = [0.5, 9.5, 9.0, 2.0, 0.0, 2.0, 0.0];
        for i in 0..7 {
            expected_closeness[i] = total_path_length[i] as f64 / 6.0;
        }

        assert_eq!(betweenness, expected_betweenness);
        assert_eq!(closeness, expected_closeness);
    }

    // Helper function to create a sample from a json file.
    // The file will begin like this:
    //   {"indices":[[2630,3217,1608,1035,...
    // and end like this:
    //   ...2316,1068,1238,704,2013]]}
    pub fn load_sample(filepath: &str) -> Sample {
        let jstring = fs::read_to_string(filepath).unwrap();
        let sample: Sample = serde_json::from_str(&jstring).unwrap();
        sample
    }

    #[test]
    #[ignore = "takes a while to run"]
    fn loaded_sample_graph() {
        let sample = load_sample("testdata/sample.json");

        // graph 1 uses integers as node value
        let mut graph1 = Graph::new();
        let mut n = 0;
        for node in &sample.indices {
            for connection in node {
                if *connection > n {
                    graph1.insert(Edge::new(n, *connection));
                }
            }
            n += 1;
        }

        let betweenness_centrality1 = graph1.betweenness_centrality(4);
        let closeness_centrality1 = graph1.closeness_centrality(4);

        // graph2 uses ip address as node value
        let mut graph2: Graph<&str> = Graph::new();
        let mut n = 0;
        for node in &sample.indices {
            for connection in node {
                if *connection > n {
                    graph2.insert(Edge::new(
                        &sample.node_ips[n],
                        &sample.node_ips[*connection],
                    ));
                }
            }
            n += 1;
        }

        // passing in zero as num_threads will be clamped to 1 thread
        let betweenness_centrality2 = graph2.betweenness_centrality(0);
        let closeness_centrality2 = graph2.closeness_centrality(0);
        let b1 = betweenness_centrality1.get(&0).unwrap();
        let b2 = betweenness_centrality2.get("65.21.141.242").unwrap();
        let c1 = closeness_centrality1.get(&0).unwrap();
        let c2 = closeness_centrality2.get("65.21.141.242").unwrap();
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);

        // Index 1837 has betweenness 9.576638518159478e-8
        // we'll confirm it's between 0.00000009 and 0.00000010
        let b1 = betweenness_centrality1.get(&1837).unwrap();
        let b2 = betweenness_centrality2.get("85.15.179.171").unwrap();
        let c1 = closeness_centrality1.get(&1837).unwrap();
        let c2 = closeness_centrality2.get("85.15.179.171").unwrap();
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
        assert!(*b1 > 0.00000009);
        assert!(*b1 < 0.00000010);

        // these should not be equal
        let b1 = betweenness_centrality1.get(&1836).unwrap();
        let b2 = betweenness_centrality2.get("85.15.179.171").unwrap();
        assert_ne!(b1, b2);
    }

    #[test]
    fn betweenness_line_topology() {
        let (a, b, c, d) = ("a", "b", "c", "d");
        let mut graph = graph!([a, b, c, d]);

        let betweenness_centrality = graph.betweenness_centrality(2);

        assert_eq!(betweenness_centrality.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(b), Some((&b, &2.0)));
        assert_eq!(betweenness_centrality.get_key_value(c), Some((&c, &2.0)));
        assert_eq!(betweenness_centrality.get_key_value(d), Some((&d, &0.0)));
    }

    #[test]
    fn betweenness_star_topology() {
        let (a, b, c, d, e) = ("a", "b", "c", "d", "e");
        let mut graph = graph!([a, b, c], [e, b, d]);

        let betweenness_centrality = graph.betweenness_centrality(2);

        assert_eq!(betweenness_centrality.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(b), Some((&b, &6.0)));
        assert_eq!(betweenness_centrality.get_key_value(c), Some((&c, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(d), Some((&d, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(e), Some((&e, &0.0)));
    }
}

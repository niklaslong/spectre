//! A module for working with graphs.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
    ops::Sub,
    fs
};
use serde::Deserialize;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::edge::Edge;

pub type Vertex = Vec<usize>;
pub type AGraph = Vec<Vec<usize>>;

#[derive(Default, Clone, Deserialize)]
pub struct AGraphSample {
    pub agraph: AGraph
}



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
        }
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

    pub fn create_agraph(&self, addresses: &Vec<T>) -> AGraph {
        let num_nodes = addresses.len();
        let mut agraph: AGraph = AGraph::new();
        for _ in 0..num_nodes {
            agraph.push(Vertex::new());
        }

        // For all our edges, check if the nodes are in our address list
        // We use the value of the addresses to find the index
        // From then on, it's all integer indices for us
        for edge in self.edges.iter() {
            let source = *edge.source();
            let target = *edge.target();

            let src_result = addresses.iter().position(|&r| r == source);
            if src_result == None {
                continue;
            }

            let tgt_result = addresses.iter().position(|&r| r == target);
            if tgt_result == None {
                continue;
            }

            let src_index = src_result.unwrap();
            let tgt_index = tgt_result.unwrap();
            agraph[src_index].push(tgt_index);
            agraph[tgt_index].push(src_index);
        }
        agraph

    }

    pub fn load_agraph(&self, crawler_report_path: &str) -> AGraph {
        let jstring = fs::read_to_string(crawler_report_path).unwrap();
        let agraph_sample: AGraphSample = serde_json::from_str(&jstring).unwrap();
        let agraph = agraph_sample.agraph;
        agraph
    }

    pub fn compute_betweenness_and_closeness (&self, agraph: &AGraph) ->  (Vec<u32>, Vec<f64>) {
        let num_nodes = agraph.len();

        let mut betweenness: Vec<u32> = vec!(0; num_nodes);
        let mut closeness: Vec<f64> = vec!(0.0; num_nodes);
        let mut total_path_length: Vec<u32> = vec!(0; num_nodes);
        let mut num_paths: Vec<u32> = vec!(0; num_nodes);

        for i in 0..num_nodes-1 {
            let mut visited: Vec<bool> = vec!(false; num_nodes);
            let mut found_or_not: Vec<bool> = vec!(false; num_nodes);
            let mut search_list: Vec<usize> = Vec::new();

            // mark node i and all those before i as visited
            for j in 0..i+1 {
                found_or_not[j] = true;
            }
            for j in i+1..num_nodes {
                search_list.push(j);
                found_or_not[j] = false;
            }

            while search_list.len() > 0 {
                // 0. OUR MAIN SEARCH LOOP:  I and J
                // 1. we search for path between i and j.  We're done when we find j
                // 2. any short paths we find along the way, they get handled and removed from search list
                // 3. along the way, we appropriately mark any between nodes
                // 4. we also determine if no path exists (disconnected graph case)
                let mut done = false;
                let j = search_list[0];
                for x in 0..num_nodes {
                    visited[x] = x == i;
                }
                let mut pathlen: u32 = 1;
                let mut queue_list = Vec::new();
                queue_list.push(i);

                while !done {
                    let mut this_round_found: Vec<usize> = Vec::new();
                    let mut queue_me = Vec::new();
                    let mut touched: bool = false;
                    for q in queue_list.as_slice() {
                        let vertex = &agraph[*q];
                        for x in vertex {
                            // We collect all shortest paths for this length, as there may be multiple paths
                            if !visited[*x] {
                                touched = true;
                                queue_me.push(*x);
                                if !found_or_not[*x] {
                                    this_round_found.push(*x);
                                    if pathlen > 1 {
                                        betweenness[*q] = betweenness[*q] + 1;
                                    }
                                }
                            }
                        }
                    }

                    queue_list.clear();
                    for x in queue_me {
                        queue_list.push(x);
                        visited[x] = true;
                    }
                    for f in this_round_found {
                        num_paths[f] = num_paths[f] + 1;
                        total_path_length[f] = total_path_length[f] + pathlen;
                        num_paths[i] = num_paths[i] + 1;
                        total_path_length[i] = total_path_length[i] + pathlen;
                        search_list.retain(|&x| x != f);
                        found_or_not[f] = true;
                        if f == j {
                            done = true;
                        }
                    }
                    // If no connection exists, stop searching for it.
                    if !touched {
                        search_list.retain(|&x| x != j);
                        found_or_not[j] = true;
                        done = true
                    }

                    pathlen = pathlen + 1;
                }
            }
        }

        for i in 0..num_nodes {
            closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
        }

        (betweenness, closeness)

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
    use std::time::Instant;

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
    fn randomish_graph() {
        let (s0, s1, s2, s3, s4, s5, s6) = ("0", "1", "2", "3", "4", "5", "6");
        let addresses = vec!["0", "1", "2", "3", "4", "5", "6"];
        let mut graph: Graph<&str> = Graph::new();
        // this graph reproduces the image at:
        // https://www.sotr.blog/articles/breadth-first-search
        graph.insert(Edge::new(s0, s3));
        graph.insert(Edge::new(s0, s5));
        graph.insert(Edge::new(s5, s1));
        graph.insert(Edge::new(s1, s2));
        graph.insert(Edge::new(s2, s4));
        graph.insert(Edge::new(s2, s6));
        graph.insert(Edge::new(s1, s3));
        let agraph = graph.create_agraph(&addresses);
        let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);

        let total_path_length = [28, 11, 13, 14, 19, 14, 19];
        let num_paths = [10, 7, 7, 7, 7, 7, 7];
        let mut expected_closeness: [f64; 7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..7 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
        }
        assert_eq!(betweenness, [1, 6, 10, 1, 0, 1, 0]);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn star_graph_a() {
        // 7-pointed star, 8 nodes
        // center is 0
        let (s0, s1, s2, s3, s4, s5, s6, s7) = ("0", "1", "2", "3", "4", "5", "6", "7");
        let addresses = vec!["0", "1", "2", "3", "4", "5", "6", "7"];
        let mut graph: Graph<&str> = Graph::new();
        graph.insert(Edge::new(s0, s1));
        graph.insert(Edge::new(s0, s2));
        graph.insert(Edge::new(s0, s3));
        graph.insert(Edge::new(s0, s4));
        graph.insert(Edge::new(s0, s5));
        graph.insert(Edge::new(s0, s6));
        graph.insert(Edge::new(s0, s7));
        let agraph = graph.create_agraph(&addresses);
        let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);

        let total_path_length = [7, 13, 13, 13, 13, 13, 13, 13];
        let num_paths = [7, 7, 7, 7, 7, 7, 7, 7];
        let mut expected_closeness: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..8 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
        }
        assert_eq!(betweenness, [21, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn star_graph_b() {
        // 7-pointed star, 8 nodes
        // center is 7
        let (s0, s1, s2, s3, s4, s5, s6, s7) = ("0", "1", "2", "3", "4", "5", "6", "7");
        let addresses = vec!["0", "1", "2", "3", "4", "5", "6", "7"];
        let mut graph: Graph<&str> = Graph::new();
        graph.insert(Edge::new(s0, s7));
        graph.insert(Edge::new(s1, s7));
        graph.insert(Edge::new(s2, s7));
        graph.insert(Edge::new(s3, s7));
        graph.insert(Edge::new(s4, s7));
        graph.insert(Edge::new(s5, s7));
        graph.insert(Edge::new(s6, s7));

        let agraph = graph.create_agraph(&addresses);
        let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);

        let total_path_length = [13, 13, 13, 13, 13, 13, 13, 7];
        let num_paths = [7, 7, 7, 7, 7, 7, 7, 7];
        let mut expected_closeness: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..8 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
        }
        assert_eq!(betweenness, [0, 0, 0, 0, 0, 0, 0, 21]);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn disconnected_graph() {
        // 9 vertices
        // 4 verts, 0-3: square, all points connected
        // 5 verts, 4-8: star, with v4 in the center
        let (s0, s1, s2, s3, s4, s5, s6, s7, s8) = ("0", "1", "2", "3", "4", "5", "6", "7", "8");
        let addresses = vec!["0", "1", "2", "3", "4", "5", "6", "7", "8"];
        let mut graph: Graph<&str> = Graph::new();
        graph.insert(Edge::new(s0, s1));
        graph.insert(Edge::new(s0, s2));
        graph.insert(Edge::new(s0, s3));
        graph.insert(Edge::new(s1, s2));
        graph.insert(Edge::new(s1, s3));
        graph.insert(Edge::new(s2, s3));

        graph.insert(Edge::new(s4, s5));
        graph.insert(Edge::new(s4, s6));
        graph.insert(Edge::new(s4, s7));
        graph.insert(Edge::new(s4, s8));

        let agraph = graph.create_agraph(&addresses);
        let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);

        let total_path_length = [3, 3, 3, 3, 4, 7, 7, 7, 7];
        let num_paths = [3, 3, 3, 3, 4, 4, 4, 4, 4];
        let mut expected_closeness: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..9 {
            expected_closeness[i] = total_path_length[i] as f64 / num_paths[i] as f64;
        }
        assert_eq!(betweenness, [0, 0, 0, 0, 6, 0, 0, 0, 0]);
        assert_eq!(closeness, expected_closeness);
    }

    #[test]
    fn imported_sample_3226() {
        let graph: Graph<usize> = Graph::new();
        let agraph = graph.load_agraph("testdata/agraph-3226.txt");
        assert_eq!(agraph.len(), 3226);
        let graph: Graph<usize> = Graph::new();
        let start = Instant::now();
        let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);
        let elapsed = start.elapsed();
        println!("elapsed for 3226 nodes: {:?}", elapsed);
        assert!(elapsed.as_secs() < 45);
        assert_eq!(agraph.len(), betweenness.len());
        assert_eq!(agraph.len(), closeness.len());
    }

    #[test]
    fn imported_sample_4914() {
        let graph: Graph<usize> = Graph::new();
        let agraph = graph.load_agraph("testdata/agraph-4914.txt");
        assert_eq!(agraph.len(), 4914);
        // this test lasts 10-11 minutes; for the time being, we skip it
        // let graph: Graph<usize> = Graph::new();
        // let start = Instant::now();
        // let (betweenness, closeness) = graph.compute_betweenness_and_closeness(&agraph);
        // let elapsed = start.elapsed();
        // println!("elapsed for 4914 nodes: {:?}", elapsed);
        // assert!(elapsed.as_secs() < 900);
        // assert_eq!(agraph.len(), betweenness.len());
        // assert_eq!(agraph.len(), closeness.len());
    }

}

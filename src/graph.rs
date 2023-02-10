//! A module for working with graphs.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    ops::Sub,
};

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::edge::Edge;

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
    T: Copy + Eq + Hash + Ord + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Graph<T>
where
    Edge<T>: Eq + Hash,
    T: Copy + Eq + Hash + Ord + Debug,
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

        if self.index.is_none() {
            self.generate_index();
        }

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

    pub fn betweenness_centrality(&mut self) -> HashMap<T, f64> {
        // B(v) = sum (shortest paths between s and t through v / total num of shortest paths
        // between s and t)
        //
        // Two other implementation options:
        //
        // 1. [Kadabra](https://drops.dagstuhl.de/opus/volltexte/2016/6371/pdf/LIPIcs-ESA-2016-20.pdf)
        // 2. [Brandes](https://pdodds.w3.uvm.edu/research/papers/others/2001/brandes2001a.pdf)

        if self.index.is_none() {
            self.generate_index();
        }

        // SAFETY: the index has already been generated in the previous block.
        let index = self.index.as_ref().unwrap();

        let nodes: Vec<usize> = index.values().copied().collect();
        let pairs: Vec<(usize, usize)> = index.values().copied().tuple_combinations().collect();

        // For each pair of nodes in the graph, compute the shortest paths.
        let mut shortest_paths = HashMap::new();

        for node in &nodes {
            for (source, target) in &pairs {
                if *source == *node && *target == *node {
                    continue;
                }

                shortest_paths.insert((*source, *target), self.shortest_paths(*source, *target));
            }
        }

        // For each shortest path between s and t, count how many go through v.
        let mut paths_through_v = HashMap::new();

        for ((source, target), paths) in &shortest_paths {
            for path in paths {
                for node in path {
                    if source == node || target == node {
                        continue;
                    }

                    paths_through_v
                        .entry((source, target, node))
                        .and_modify(|e| *e += 1.0f64)
                        .or_insert(1.0);
                }
            }
        }

        // Calculate the centrality for each node.
        let mut centralities = HashMap::new();

        // SAFETY: the index has already been generated.
        for (n, i) in self.index.as_ref().unwrap() {
            let centrality: f64 = paths_through_v
                .iter()
                .filter(|((_source, _target, node), _count)| i == *node)
                .map(|((source, target, _node), count)| {
                    // Shortest paths between s and t through v divided by the total number of
                    // shortest paths between s and t.
                    //
                    // SAFETY: the paths must exist and there is no division by zero.
                    *count / shortest_paths.get(&(**source, **target)).unwrap().len() as f64
                })
                .sum();

            centralities.insert(*n, centrality);
        }

        centralities
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

    fn shortest_paths(&mut self, source: usize, target: usize) -> Vec<Vec<usize>> {
        let adjacency_matrix = self.adjacency_matrix();

        let mut visited = vec![false; adjacency_matrix.nrows()];
        visited[0] = true;

        // Each index contains a vector of paths (each path is a vector of usize).
        let mut paths: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
        paths.insert(source, vec![vec![source]]);

        // At each "layer" we determine what layers to search next (i.e. rows).
        let mut layer = VecDeque::new();
        let mut next_layer = VecDeque::new();

        // Start the search at the source.
        layer.push_back(source);

        // Indexes used are M x N, i.e. M = row index, N = column index.
        while let Some(m) = layer.pop_front() {
            // Find all the indexes of the vertices connected to the current search source.
            let neighbours: Vec<usize> = adjacency_matrix
                .row(m)
                .iter()
                .enumerate()
                .filter(|(n, &val)| val == 1.0 && !visited[*n])
                .map(|(n, _)| n)
                .collect();

            // Remember where we've been so we don't accidentally backtrack. Exclude the target so
            // all shortest paths can be found.
            for n in neighbours {
                if n != target {
                    visited[n] = true;

                    // Make sure the source or the target are never included in the next layer.
                    if n != source {
                        next_layer.push_back(n);
                    }
                }

                // Fetch the paths that end in the current search source (m).
                // SAFETY: entry must exist as it is set to the source at the start of the search.
                let m_paths = paths.get(&m).unwrap();
                let mut n_paths = vec![];

                // Extend each "m"-ending path with "n" to create the "n"-ending paths.
                for m_path in m_paths {
                    let mut n_path: Vec<usize> = vec![];
                    n_path.extend_from_slice(m_path);
                    n_path.push(n);

                    n_paths.push(n_path);
                }

                // Append the newest shortest paths ending in the target to ones found previously.
                paths
                    .entry(n)
                    .and_modify(|e| e.append(&mut n_paths))
                    .or_insert(n_paths);
            }

            // If a layer has been fully searched, switch to the next layer only if the target
            // hasn't been found at this depth. The next layer will be empty if the target has been
            // found.
            if layer.is_empty() {
                layer.append(&mut next_layer)
            }
        }

        // If we're done, retrieve the target paths.
        paths.remove(&target).unwrap()
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
    fn one_shortest_path() {
        let (a, b, c) = ("a", "b", "c");
        let mut graph = graph!([a, b, c]);

        // Indexing corresponds to naming: a: 0, b: 1, c: 2.
        assert_eq!(graph.shortest_paths(0, 2), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn two_shortest_paths() {
        let (a, b, c, d) = ("a", "b", "c", "d");
        let mut graph = graph!([a, b, c], [a, d, c]);

        assert_eq!(
            graph.shortest_paths(0, 2),
            vec![vec![0, 1, 2], vec![0, 3, 2]]
        );
    }

    #[test]
    fn ignore_longer_paths() {
        let (a, b, c, d, e) = ("a", "b", "c", "d", "e");
        let mut graph = graph!([a, b, c], [a, d, c], [a, e, d, c]);

        assert_eq!(
            graph.shortest_paths(0, 2),
            vec![vec![0, 1, 2], vec![0, 3, 2]]
        );
    }

    #[test]
    fn no_duplicate_paths() {
        let (a, b, c, d) = ("a", "b", "c", "d");
        let mut graph = graph!([a, b, c, d]);

        assert_eq!(graph.shortest_paths(2, 3), vec![vec![2, 3]])
    }

    #[test]
    fn betweenness() {
        let (a, b, c, d) = ("a", "b", "c", "d");
        let mut graph = graph!([a, b, c, d]);

        let betweenness_centrality = graph.betweenness_centrality();

        assert_eq!(betweenness_centrality.get_key_value(a), Some((&a, &0.0)));
        assert_eq!(betweenness_centrality.get_key_value(b), Some((&b, &2.0)));
        assert_eq!(betweenness_centrality.get_key_value(c), Some((&c, &2.0)));
        assert_eq!(betweenness_centrality.get_key_value(d), Some((&d, &0.0)));
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
}

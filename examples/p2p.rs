use std::{collections::HashSet, net::SocketAddr};

use spectre::{edge::Edge, graph::Graph};

// A mock node implementation, has an address and stores peers.
struct Node {
    addr: SocketAddr,
    peers: HashSet<SocketAddr>,
}

impl Node {
    fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            peers: HashSet::new(),
        }
    }

    fn addr(&self) -> &SocketAddr {
        &self.addr
    }

    fn peers(&self) -> &HashSet<SocketAddr> {
        &self.peers
    }

    fn connect(&mut self, peer: &mut Node) {
        self.peers.insert(*peer.addr());
        peer.peers.insert(*self.addr());
    }

    fn disconnect(&mut self, peer: &mut Node) {
        self.peers.remove(peer.addr());
        peer.peers.remove(self.addr());
    }
}

fn main() {
    const N: usize = 3;
    let mut nodes: Vec<Node> = vec![];

    for i in 0..N {
        let mut node = Node::new(format!("127.0.0.1:000{i}").parse().unwrap());

        // For each node connect to the previous node as a peer, creating a line topology.
        if let Some(peer) = nodes.last_mut() {
            node.connect(peer);
        }

        nodes.push(node)
    }

    println!("\nCrawling network with {} nodes...", N);

    // Simulate crawling the network and use each node's peers to create the graph.
    let mut graph = Graph::new();
    for node in &nodes {
        for peer in node.peers() {
            graph.insert(Edge::new(*node.addr(), *peer));
        }
    }

    println!(
        "Total connection count: {}, adjacency matrix: {}",
        graph.edge_count(),
        graph.adjacency_matrix()
    );

    // Update the topology by connecting the two bounding nodes in the set.
    println!("Connecting first and last node...");
    let mut last = nodes.pop().expect("vec has no last item");

    nodes
        .first_mut()
        .expect("vec has no first item")
        .connect(&mut last);

    // Add the node back into the set after mutating it.
    nodes.push(last);

    // Simulate crawling the network and updating the graph.
    for node in &nodes {
        let peers: Vec<SocketAddr> = node.peers.iter().cloned().collect();
        graph.update_subset(*node.addr(), &peers);
    }

    println!(
        "Total connection count: {}, adjacency matrix: {}",
        graph.edge_count(),
        graph.adjacency_matrix()
    );

    // Update the topology by removing a connection.
    println!("Disconnecting second to last and last node...");
    let mut last = nodes.pop().expect("vec has no last item");
    let mut second_to_last = nodes.pop().expect("vec has no last item");

    last.disconnect(&mut second_to_last);

    // Add the nodes back into the set after mutating it.
    nodes.push(second_to_last);
    nodes.push(last);

    // Simulate crawling the network and updating the graph.
    for node in nodes {
        let peers: Vec<SocketAddr> = node.peers.iter().cloned().collect();
        graph.update_subset(*node.addr(), &peers);
    }

    println!(
        "Total connection count: {}, adjacency matrix: {}",
        graph.edge_count(),
        graph.adjacency_matrix()
    );
}

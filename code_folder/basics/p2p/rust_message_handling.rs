use libp2p::{
    identity::Keypair,
    swarm::{NetworkBehaviour, NetworkBehaviourEventProcess, Swarm},
    Multiaddr,
};
use libp2p::ping::Ping;
use libp2p::kad::{Kademlia, KademliaEvent};
use tokio::net::TcpListener;

#[derive(NetworkBehaviour)]
struct MyBehaviour {
    ping: Ping,
    kademlia: Kademlia<libp2p::kad::record::store::MemoryStore>,
}

impl MyBehaviour {
    fn new(peer_id: libp2p::PeerId) -> Self {
        let store = libp2p::kad::record::store::MemoryStore::new(peer_id);
        let kademlia = Kademlia::new(peer_id, store);
        let ping = Ping::new();
        MyBehaviour { ping, kademlia }
    }
}

impl NetworkBehaviourEventProcess<libp2p::ping::PingEvent> for MyBehaviour {
    fn inject_event(&mut self, event: libp2p::ping::PingEvent) {
        match event {
            libp2p::ping::PingEvent::PingReceived { peer, rtt } => {
                println!("Received ping from {:?} with RTT {:?}", peer, rtt);
            },
            _ => {},
        }
    }
}

impl NetworkBehaviourEventProcess<KademliaEvent> for MyBehaviour {
    fn inject_event(&mut self, event: KademliaEvent) {
        match event {
            KademliaEvent::Discovered { peer_id } => {
                println!("Discovered peer: {:?}", peer_id);
                self.ping.ping(peer_id);
            },
            KademliaEvent::Connected { peer_id } => {
                println!("Connected to peer: {:?}", peer_id);
                self.ping.ping(peer_id);
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    let local_key = Keypair::generate_ed25519();
    let local_peer_id = local_key.public().into_peer_id();
    let transport = libp2p::build_tcp_ws_secio_yamux(local_key);
    let mut swarm = {
        let behaviour = MyBehaviour::new(local_peer_id);
        Swarm::new(transport, behaviour, local_peer_id)
    };

    let listen_addr = "/ip4/0.0.0.0/tcp/0".parse::<Multiaddr>().unwrap();
    Swarm::listen_on(&mut swarm, listen_addr).unwrap();

    let mut listening = false;
    tokio::signal::ctrl_c().await.

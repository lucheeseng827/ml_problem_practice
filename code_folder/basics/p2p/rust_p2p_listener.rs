use libp2p::{
    identity::Keypair,
    swarm::{NetworkBehaviour, NetworkBehaviourEventProcess, Swarm},
    Multiaddr,
};

use tokio::net::TcpListener;

struct MyBehaviour;

impl NetworkBehaviour for MyBehaviour {
    type ProtocolsHandler = ();
    type OutEvent = ();

    fn new_handler(&mut self) -> Self::ProtocolsHandler {
        ()
    }

    fn addresses_of_peer(&mut self, _peer_id: &libp2p::PeerId) -> Vec<Multiaddr> {
        Vec::new()
    }

    fn inject_connected(&mut self, _peer_id: libp2p::PeerId) {
        println!("Connected to a new peer: {:?}", _peer_id);
    }

    fn inject_disconnected(&mut self, _peer_id: libp2p::PeerId) {
        println!("Disconnected from a peer: {:?}", _peer_id);
    }

    fn inject_event(
        &mut self,
        _peer_id: libp2p::PeerId,
        _event: <Self::ProtocolsHandler as libp2p::ProtocolsHandler>::OutEvent,
    ) {
    }
}

#[tokio::main]
async fn main() {
    let local_key = Keypair::generate_ed25519();
    let local_peer_id = local_key.public().into_peer_id();
    let transport = libp2p::build_tcp_ws_secio_yamux(local_key);
    let mut swarm = {
        let mut behaviour = MyBehaviour;
        Swarm::new(transport, behaviour, local_peer_id)
    };

    let listen_addr = "/ip4/0.0.0.0/tcp/0".parse::<Multiaddr>().unwrap();
    Swarm::listen_on(&mut swarm, listen_addr).unwrap();

    let mut listening = false;
    tokio::signal::ctrl_c().await.unwrap();
    Swarm::close(&mut swarm).await;
}

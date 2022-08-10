mod neu;
use rand;
fn main() {
    let mut net = neu::NeurNet::new(0.7, 0.3, vec![2, 2, 1]);

    let study = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

    for i in 0..1000000 {
        let mut er = 0 as f64;
        for s in study {
            net.define(vec![s[0], s[1]]);
            net.calc();
            er += net.estimate(s[2]);
            net.tune(s[2]);
        }
        print!("\r{}:{}", i, er / 4.0);
    }
    println!("");
}

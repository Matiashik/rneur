use rand;

fn err(exp: f64, res: f64) -> f64 {
    return (exp - res).powi(2);
}

fn sigm(x: f64) -> f64 {
    return 1.0 / (1.0 + std::f64::consts::E.powf(-x));
}

fn dsigm(x: f64) -> f64 {
    return (1.0 - x) * x;
}

#[derive(Debug)]
pub struct Neur {
    pub val: f64,
    pub delta: f64,
}

impl Neur {
    fn new(val: f64, delta: f64) -> Neur {
        return Neur {
            val: val,
            delta: delta,
        };
    }
}

#[derive(Debug)]
pub struct Connection {
    pub inp: u16,
    pub out: u16,
    pub w: f64,
    pub dw: f64,
}

impl Connection {
    fn new(inp: u16, out: u16, w: f64, dw: f64) -> Connection {
        return Connection {
            inp: inp,
            out: out,
            w: w,
            dw: dw,
        };
    }
}

#[derive(Debug)]
pub struct NeurNet {
    pub neur_lyr: Vec<Vec<Neur>>,
    pub con_lyr: Vec<Vec<Connection>>,
    pub n: f64,
    pub a: f64,
}

impl NeurNet {
    pub fn new(n: f64, a: f64, shape: Vec<u16>) -> NeurNet {
        let mut nl: Vec<Vec<Neur>> = Vec::new();
        for i in 0..shape.len() {
            let mut t: Vec<Neur> = Vec::new();
            for _ in 0..shape[i] {
                t.push(Neur::new(0.0, 0.0));
            }
            nl.push(t)
        }
        let mut cl: Vec<Vec<Connection>> = Vec::new();
        for i in 0..shape.len() - 1 {
            let mut t: Vec<Connection> = Vec::new();
            for j in 0..shape[i] {
                for k in 0..shape[i + 1] {
                    t.push(Connection::new(j, k, rand::random::<f64>(), 0.0));
                }
            }
            cl.push(t)
        }
        return NeurNet {
            n: n,
            a: a,
            neur_lyr: nl,
            con_lyr: cl,
        };
    }

    pub fn define(&mut self, input: Vec<f64>) {
        for i in 0..self.neur_lyr[0].len() {
            self.neur_lyr[0][i].val = sigm(input[i]);
        }
    }

    pub fn calc(&mut self) -> Vec<f64> {
        for lyr in 1..self.neur_lyr.len() {
            for ner in 0..self.neur_lyr[lyr].len() {
                self.neur_lyr[lyr][ner].val = 0.0;
                for con in 0..self.con_lyr[lyr - 1].len() {
                    if self.con_lyr[lyr - 1][con].out as usize == ner {
                        self.neur_lyr[lyr][ner].val +=
                            self.neur_lyr[lyr - 1][self.con_lyr[lyr - 1][con].inp as usize].val
                                * self.con_lyr[lyr - 1][con].w;
                    }
                }
                self.neur_lyr[lyr][ner].val = sigm(self.neur_lyr[lyr][ner].val);
            }
        }
        let mut res: Vec<f64> = Vec::new();
        for i in 0..self.neur_lyr.last().unwrap().len() {
            res.push(self.neur_lyr.last().unwrap()[i].val);
        }
        return res;
    }

    pub fn estimate(&self, expected: Vec<f64>) -> Vec<f64> {
        let mut res: Vec<f64> = Vec::new();
        for i in 0..expected.len() {
            res.push(err(expected[i], self.neur_lyr.last().unwrap()[i].val));
        }
        return res;
    }

    pub fn tune(&mut self, expected: Vec<f64>) {
        let ln = self.neur_lyr.len();
        for ner in 0..self.neur_lyr[ln - 1].len() {
            self.neur_lyr[ln - 1][ner].delta = (expected[ner]
                - self.neur_lyr.last().unwrap()[ner].val)
                * dsigm(self.neur_lyr.last().unwrap()[ner].val);
        }
        for lyr in (0..ln - 1).rev() {
            for ner in 0..self.neur_lyr[lyr].len() {
                self.neur_lyr[lyr][ner].delta = 0.0;
                for con in 0..self.con_lyr[lyr].len() {
                    if self.con_lyr[lyr][con].inp as usize == ner {
                        self.neur_lyr[lyr][ner].delta += self.con_lyr[lyr][con].w
                            * self.neur_lyr[lyr + 1][self.con_lyr[lyr][con].out as usize].delta;
                    }
                }
                self.neur_lyr[lyr][ner].delta =
                    dsigm(self.neur_lyr[lyr][ner].val) * self.neur_lyr[lyr][ner].delta;
            }
            for con in 0..self.con_lyr[lyr].len() {
                let grad = self.neur_lyr[lyr][self.con_lyr[lyr][con].inp as usize].val
                    * self.neur_lyr[lyr + 1][self.con_lyr[lyr][con].out as usize].delta;
                self.con_lyr[lyr][con].dw = self.n * grad + self.a * self.con_lyr[lyr][con].dw;
                self.con_lyr[lyr][con].w += self.con_lyr[lyr][con].dw;
            }
        }
    }
}

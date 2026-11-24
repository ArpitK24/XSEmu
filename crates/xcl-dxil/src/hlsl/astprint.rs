//! The AST back as text (owner: the parser subagent).
//!
//! [`unit`] writes HLSL that parses to the same tree: every operator application in its own
//! parentheses, so nothing depends on precedence, macros and `#if` already resolved. It is
//! how the tests check the tree against DXC (the printed program must compile to the same
//! DXIL as the original) and against the parser itself (`parse(unit(parse(x)))` is
//! `parse(x)`). [`tree`] writes every node as an S-expression with names instead of
//! numbers: two units are the same tree exactly when their dumps are equal (their ids need
//! not be: they depend on the order names and types first appear, which printing can
//! change).

use std::fmt::Write;

use super::ast::*;
use super::lex::{res_name, scalar_name, QUALS};

/// The unit as HLSL source.
pub fn unit(u: &Unit) -> String {
    let mut p = Printer { u, out: String::with_capacity(u.exprs.len() * 12 + 256), indent: 0 };
    for g in &u.globals {
        match *g {
            Global::Struct(i) => {
                let s = &u.structs[i as usize];
                let _ = writeln!(p.out, "struct {} {{", u.name(s.name));
                for f in &s.fields {
                    p.out.push_str("    ");
                    p.var(f, true);
                    p.out.push_str(";\n");
                }
                p.out.push_str("};\n");
            }
            Global::Var(i) => {
                p.var(&u.vars[i as usize], true);
                p.out.push_str(";\n");
            }
            Global::Func(i) => p.func(&u.funcs[i as usize]),
        }
    }
    p.out
}

/// The unit as S-expressions: all of the tree except line numbers and ids.
pub fn tree(u: &Unit) -> String {
    let mut d = Dump { p: Printer { u, out: String::with_capacity(u.exprs.len() * 16 + 256), indent: 0 } };
    for g in &u.globals {
        match *g {
            Global::Struct(i) => {
                let s = &u.structs[i as usize];
                let _ = write!(d.p.out, "(struct {}", u.name(s.name));
                for f in &s.fields {
                    d.p.out.push(' ');
                    d.var(f);
                }
                d.p.out.push_str(")\n");
            }
            Global::Var(i) => {
                d.var(&u.vars[i as usize]);
                d.p.out.push('\n');
            }
            Global::Func(i) => {
                let f = &u.funcs[i as usize];
                let _ = write!(d.p.out, "(func {} ", u.name(f.name));
                d.p.ty(f.ret);
                if let Some(s) = f.ret_semantic {
                    let _ = write!(d.p.out, " :{}", u.name(s));
                }
                for a in &f.attrs {
                    d.p.out.push(' ');
                    d.p.attr(a);
                }
                d.p.out.push_str(" (params");
                for v in &f.params {
                    d.p.out.push(' ');
                    d.var(v);
                }
                d.p.out.push(')');
                match f.body {
                    Some(b) => {
                        d.p.out.push('\n');
                        d.stmts(b);
                    }
                    None => d.p.out.push_str(" prototype"),
                }
                d.p.out.push_str(")\n");
            }
        }
    }
    d.p.out
}

struct Printer<'a> {
    u: &'a Unit,
    out: String,
    indent: usize,
}

fn unary_text(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Plus => "+",
        UnOp::Not => "!",
        UnOp::BitNot => "~",
        UnOp::PreInc | UnOp::PostInc => "++",
        UnOp::PreDec | UnOp::PostDec => "--",
    }
}

fn binary_text(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::LogAnd => "&&",
        BinOp::LogOr => "||",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::Comma => ",",
    }
}

impl Printer<'_> {
    fn name(&mut self, s: Sym) {
        self.out.push_str(self.u.name(s));
    }

    fn ty(&mut self, t: TypeId) {
        match *self.u.ty(t) {
            TypeRef::Void => self.out.push_str("void"),
            TypeRef::Num { kind, rows, cols } => {
                self.out.push_str(scalar_name(kind));
                if rows > 1 {
                    let _ = write!(self.out, "{rows}x{cols}");
                } else if cols > 1 {
                    let _ = write!(self.out, "{cols}");
                }
            }
            TypeRef::Named(s) => self.name(s),
            TypeRef::Resource { kind, elem, samples } => {
                self.out.push_str(res_name(kind));
                if let Some(e) = elem {
                    self.out.push('<');
                    self.ty(e);
                    if let Some(n) = samples {
                        let _ = write!(self.out, ", {n}");
                    }
                    self.close_template();
                }
            }
            TypeRef::Sampler { comparison } => self.out.push_str(if comparison { "SamplerComparisonState" } else { "SamplerState" }),
            TypeRef::Stream { kind, elem } => {
                self.out.push_str(match kind {
                    StreamKind::Point => "PointStream<",
                    StreamKind::Line => "LineStream<",
                    StreamKind::Triangle => "TriangleStream<",
                });
                self.ty(elem);
                self.close_template();
            }
            TypeRef::Patch { output, elem, points } => {
                self.out.push_str(if output { "OutputPatch<" } else { "InputPatch<" });
                self.ty(elem);
                let _ = write!(self.out, ", {points}");
                self.close_template();
            }
        }
    }

    fn close_template(&mut self) {
        // never `>>`
        if self.out.ends_with('>') {
            self.out.push(' ');
        }
        self.out.push('>');
    }

    fn quals(&mut self, q: u32) {
        if q & qual::INOUT == qual::INOUT {
            self.out.push_str("inout ");
        }
        for (word, bits, _) in QUALS {
            let in_out = bits & qual::INOUT != 0;
            if bits != qual::INOUT && q & bits == bits && !(in_out && q & qual::INOUT == qual::INOUT) {
                self.out.push_str(word);
                self.out.push(' ');
            }
        }
    }

    /// A declaration without its `;`; `typed`: with qualifiers and type (the first
    /// declarator of a statement).
    fn var(&mut self, v: &VarDecl, typed: bool) {
        if typed {
            self.quals(v.quals);
            self.ty(v.ty);
            self.out.push(' ');
        }
        self.name(v.name);
        for &d in &v.dims {
            self.out.push('[');
            self.expr(d);
            self.out.push(']');
        }
        if let Some(s) = v.semantic {
            self.out.push_str(" : ");
            self.name(s);
        }
        if let Some(r) = v.register {
            let _ = write!(self.out, " : register({}{}, space{})", r.class as char, r.index, r.space);
        }
        if let Some(i) = v.init {
            self.out.push_str(" = ");
            self.expr(i);
        }
    }

    fn attr(&mut self, a: &Attr) {
        self.out.push('[');
        self.name(a.name);
        if !a.args.is_empty() {
            self.out.push('(');
            for (i, arg) in a.args.iter().enumerate() {
                if i != 0 {
                    self.out.push_str(", ");
                }
                match arg {
                    AttrArg::Int(v) => {
                        let _ = write!(self.out, "{v}");
                    }
                    AttrArg::Float(v) => self.float(*v, FloatSuffix::None),
                    AttrArg::Str(s) => {
                        let _ = write!(self.out, "\"{s}\"");
                    }
                    AttrArg::Ident(s) => self.name(*s),
                }
            }
            self.out.push(')');
        }
        self.out.push(']');
    }

    fn func(&mut self, f: &Func) {
        for a in &f.attrs {
            self.attr(a);
            self.out.push('\n');
        }
        self.ty(f.ret);
        self.out.push(' ');
        self.name(f.name);
        self.out.push('(');
        for (i, p) in f.params.iter().enumerate() {
            if i != 0 {
                self.out.push_str(", ");
            }
            self.var(p, true);
        }
        self.out.push(')');
        if let Some(s) = f.ret_semantic {
            self.out.push_str(" : ");
            self.name(s);
        }
        match f.body {
            Some(b) => {
                self.out.push('\n');
                self.block(b);
            }
            None => self.out.push_str(";\n"),
        }
    }

    fn pad(&mut self) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
    }

    fn block(&mut self, r: Range) {
        self.pad();
        self.out.push_str("{\n");
        self.indent += 1;
        for &s in self.u.block(r) {
            self.stmt(s);
        }
        self.indent -= 1;
        self.pad();
        self.out.push_str("}\n");
    }

    /// A statement nested in another: a block at the same indentation, anything else one
    /// level in.
    fn nested(&mut self, s: StmtId) {
        if matches!(self.u.stmt(s), Stmt::Block(_)) && !self.u.stmt_attrs.contains_key(&s.0) {
            self.stmt(s);
        } else {
            self.indent += 1;
            self.stmt(s);
            self.indent -= 1;
        }
    }

    /// The declarations of a [`Stmt::Var`], without the `;`.
    fn vars(&mut self, r: Range) {
        for (i, v) in r.of(&self.u.vars).iter().enumerate() {
            if i != 0 {
                self.out.push_str(", ");
            }
            self.var(v, i == 0);
        }
    }

    /// An expression where a statement could also be a declaration: in parentheses if its
    /// text would start with a type name.
    fn stmt_expr(&mut self, e: ExprId) {
        let mut first = e;
        let typed = loop {
            match *self.u.expr(first) {
                Expr::Member(o, _) | Expr::Index(o, _) => first = o,
                Expr::Method(_, all) => first = self.u.method(all).0,
                Expr::Construct(..) => break true,
                _ => break false,
            }
        };
        if typed {
            self.out.push('(');
            self.expr(e);
            self.out.push(')');
        } else {
            self.expr(e);
        }
    }

    fn stmt(&mut self, id: StmtId) {
        if let Some(attrs) = self.u.stmt_attrs.get(&id.0) {
            self.pad();
            for a in attrs {
                self.attr(a);
                self.out.push(' ');
            }
            self.out.push('\n');
        }
        let s = *self.u.stmt(id);
        if let Stmt::Block(r) = s {
            self.block(r);
            return;
        }
        self.pad();
        match s {
            Stmt::Block(_) => {}
            Stmt::Var(r) => {
                self.vars(r);
                self.out.push_str(";\n");
            }
            Stmt::Expr(e) => {
                self.stmt_expr(e);
                self.out.push_str(";\n");
            }
            Stmt::If { cond, then, els } => {
                self.out.push_str("if (");
                self.expr(cond);
                self.out.push_str(")\n");
                self.nested(then);
                if let Some(e) = els {
                    self.pad();
                    self.out.push_str("else\n");
                    self.nested(e);
                }
            }
            Stmt::For { init, cond, step, body } => {
                self.out.push_str("for (");
                match init.map(|i| *self.u.stmt(i)) {
                    Some(Stmt::Var(r)) => self.vars(r),
                    Some(Stmt::Expr(e)) => self.stmt_expr(e),
                    _ => {}
                }
                self.out.push_str("; ");
                if let Some(c) = cond {
                    self.expr(c);
                }
                self.out.push_str("; ");
                if let Some(e) = step {
                    self.expr(e);
                }
                self.out.push_str(")\n");
                self.nested(body);
            }
            Stmt::While { cond, body } => {
                self.out.push_str("while (");
                self.expr(cond);
                self.out.push_str(")\n");
                self.nested(body);
            }
            Stmt::DoWhile { body, cond } => {
                self.out.push_str("do\n");
                self.nested(body);
                self.pad();
                self.out.push_str("while (");
                self.expr(cond);
                self.out.push_str(");\n");
            }
            Stmt::Switch { value, cases } => {
                self.out.push_str("switch (");
                self.expr(value);
                self.out.push_str(")\n");
                self.pad();
                self.out.push_str("{\n");
                for c in cases.of(&self.u.cases) {
                    self.pad();
                    match c.value {
                        Some(v) => {
                            self.out.push_str("case ");
                            self.expr(v);
                            self.out.push_str(":\n");
                        }
                        None => self.out.push_str("default:\n"),
                    }
                    self.indent += 1;
                    for &s in self.u.block(c.body) {
                        self.stmt(s);
                    }
                    self.indent -= 1;
                }
                self.pad();
                self.out.push_str("}\n");
            }
            Stmt::Break => self.out.push_str("break;\n"),
            Stmt::Continue => self.out.push_str("continue;\n"),
            Stmt::Discard => self.out.push_str("discard;\n"),
            Stmt::Return(v) => {
                self.out.push_str("return");
                if let Some(v) = v {
                    self.out.push(' ');
                    self.expr(v);
                }
                self.out.push_str(";\n");
            }
            Stmt::Empty => self.out.push_str(";\n"),
        }
    }

    /// A floating-point literal that reads back as the same value: the shortest decimal
    /// that rounds to it (as an `f32` with an `f` or `h` suffix, where the value is one).
    fn float(&mut self, v: f64, suffix: FloatSuffix) {
        let at = self.out.len();
        let single = matches!(suffix, FloatSuffix::F | FloatSuffix::H);
        if !v.is_finite() {
            // only a literal beyond the range reads as infinity
            self.out.push_str(if single { "1.0e99" } else { "1.0e999" });
        } else if single {
            let _ = write!(self.out, "{:?}", v as f32);
        } else {
            let _ = write!(self.out, "{v:?}");
        }
        debug_assert!(self.out[at..].contains(['.', 'e']));
        self.out.push_str(match suffix {
            FloatSuffix::None => "",
            FloatSuffix::F => "f",
            FloatSuffix::H => "h",
            FloatSuffix::L => "l",
        });
    }

    fn args(&mut self, r: Range) {
        self.out.push('(');
        for (i, &a) in self.u.args(r).iter().enumerate() {
            if i != 0 {
                self.out.push_str(", ");
            }
            self.expr(a);
        }
        self.out.push(')');
    }

    /// The left side of `.` or `[`: a literal goes in parentheses (`1.x` is not a member
    /// access); every other expression already prints as a primary expression.
    fn base(&mut self, e: ExprId) {
        if matches!(self.u.expr(e), Expr::Int(..) | Expr::Float(..) | Expr::Bool(_) | Expr::InitList(_)) {
            self.out.push('(');
            self.expr(e);
            self.out.push(')');
        } else {
            self.expr(e);
        }
    }

    fn expr(&mut self, e: ExprId) {
        match *self.u.expr(e) {
            Expr::Int(v, suffix) => {
                let _ = write!(self.out, "{v}");
                self.out.push_str(match suffix {
                    IntSuffix::None => "",
                    IntSuffix::U => "u",
                    IntSuffix::L => "l",
                    IntSuffix::UL => "ul",
                    IntSuffix::LL => "ll",
                    IntSuffix::ULL => "ull",
                });
            }
            Expr::Float(v, suffix) => self.float(v, suffix),
            Expr::Bool(b) => self.out.push_str(if b { "true" } else { "false" }),
            Expr::Ident(s) => self.name(s),
            Expr::Unary(op, a) => {
                self.out.push('(');
                if matches!(op, UnOp::PostInc | UnOp::PostDec) {
                    self.expr(a);
                    self.out.push_str(unary_text(op));
                } else {
                    self.out.push_str(unary_text(op));
                    self.expr(a);
                }
                self.out.push(')');
            }
            Expr::Binary(op, a, b) => {
                self.out.push('(');
                self.expr(a);
                if op != BinOp::Comma {
                    self.out.push(' ');
                }
                self.out.push_str(binary_text(op));
                self.out.push(' ');
                self.expr(b);
                self.out.push(')');
            }
            Expr::Assign(op, a, b) => {
                self.out.push('(');
                self.expr(a);
                self.out.push(' ');
                if let Some(op) = op {
                    self.out.push_str(binary_text(op));
                }
                self.out.push_str("= ");
                self.expr(b);
                self.out.push(')');
            }
            Expr::Ternary(c, t, f) => {
                self.out.push('(');
                self.expr(c);
                self.out.push_str(" ? ");
                self.expr(t);
                self.out.push_str(" : ");
                self.expr(f);
                self.out.push(')');
            }
            Expr::Call(name, args) => {
                self.name(name);
                self.args(args);
            }
            Expr::Construct(t, args) => {
                self.ty(t);
                self.args(args);
            }
            Expr::Cast(t, a) => {
                self.out.push_str("((");
                self.ty(t);
                self.out.push(')');
                self.expr(a);
                self.out.push(')');
            }
            Expr::Method(name, all) => {
                let (o, args) = self.u.method(all);
                self.base(o);
                self.out.push('.');
                self.name(name);
                self.args(args);
            }
            Expr::Member(o, name) => {
                self.base(o);
                self.out.push('.');
                self.name(name);
            }
            Expr::Index(o, i) => {
                self.base(o);
                self.out.push('[');
                self.expr(i);
                self.out.push(']');
            }
            Expr::InitList(r) => {
                self.out.push_str("{ ");
                for (i, &a) in self.u.args(r).iter().enumerate() {
                    if i != 0 {
                        self.out.push_str(", ");
                    }
                    self.expr(a);
                }
                self.out.push_str(" }");
            }
        }
    }
}

struct Dump<'a> {
    p: Printer<'a>,
}

impl Dump<'_> {
    fn var(&mut self, v: &VarDecl) {
        let u = self.p.u;
        let _ = write!(self.p.out, "(var {} ", u.name(v.name));
        self.p.quals(v.quals);
        self.p.ty(v.ty);
        for &d in &v.dims {
            self.p.out.push_str(" [");
            self.expr(d);
            self.p.out.push(']');
        }
        if let Some(s) = v.semantic {
            let _ = write!(self.p.out, " :{}", u.name(s));
        }
        if let Some(r) = v.register {
            let _ = write!(self.p.out, " @{}{},{}", r.class as char, r.index, r.space);
        }
        if let Some(i) = v.init {
            self.p.out.push_str(" = ");
            self.expr(i);
        }
        self.p.out.push(')');
    }

    fn stmts(&mut self, r: Range) {
        let u = self.p.u;
        self.p.indent += 1;
        for &s in u.block(r) {
            self.stmt(s);
        }
        self.p.indent -= 1;
    }

    fn child(&mut self, s: StmtId) {
        self.p.indent += 1;
        self.stmt(s);
        self.p.indent -= 1;
    }

    fn stmt(&mut self, id: StmtId) {
        let u = self.p.u;
        self.p.pad();
        if let Some(attrs) = u.stmt_attrs.get(&id.0) {
            for a in attrs {
                self.p.attr(a);
                self.p.out.push(' ');
            }
        }
        match *u.stmt(id) {
            Stmt::Var(r) => {
                self.p.out.push_str("(decl");
                for v in r.of(&u.vars) {
                    self.p.out.push(' ');
                    self.var(v);
                }
                self.p.out.push_str(")\n");
            }
            Stmt::Expr(e) => {
                self.expr(e);
                self.p.out.push('\n');
            }
            Stmt::Block(r) => {
                self.p.out.push_str("(block\n");
                self.stmts(r);
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::If { cond, then, els } => {
                self.p.out.push_str("(if ");
                self.expr(cond);
                self.p.out.push('\n');
                self.child(then);
                if let Some(e) = els {
                    self.p.pad();
                    self.p.out.push_str(" else\n");
                    self.child(e);
                }
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::For { init, cond, step, body } => {
                self.p.out.push_str("(for\n");
                match init {
                    Some(i) => self.child(i),
                    None => {
                        self.p.pad();
                        self.p.out.push_str("    -\n");
                    }
                }
                self.p.pad();
                self.p.out.push_str("    ");
                match cond {
                    Some(c) => self.expr(c),
                    None => self.p.out.push('-'),
                }
                self.p.out.push_str(" ; ");
                match step {
                    Some(e) => self.expr(e),
                    None => self.p.out.push('-'),
                }
                self.p.out.push('\n');
                self.child(body);
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::While { cond, body } => {
                self.p.out.push_str("(while ");
                self.expr(cond);
                self.p.out.push('\n');
                self.child(body);
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::DoWhile { body, cond } => {
                self.p.out.push_str("(do ");
                self.expr(cond);
                self.p.out.push('\n');
                self.child(body);
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::Switch { value, cases } => {
                self.p.out.push_str("(switch ");
                self.expr(value);
                self.p.out.push('\n');
                for c in cases.of(&u.cases) {
                    self.p.pad();
                    match c.value {
                        Some(v) => {
                            self.p.out.push_str(" (case ");
                            self.expr(v);
                        }
                        None => self.p.out.push_str(" (default"),
                    }
                    self.p.out.push('\n');
                    self.p.indent += 1;
                    self.stmts(c.body);
                    self.p.indent -= 1;
                    self.p.pad();
                    self.p.out.push_str(" )\n");
                }
                self.p.pad();
                self.p.out.push_str(")\n");
            }
            Stmt::Break => self.p.out.push_str("break\n"),
            Stmt::Continue => self.p.out.push_str("continue\n"),
            Stmt::Discard => self.p.out.push_str("discard\n"),
            Stmt::Return(v) => {
                self.p.out.push_str("(return");
                if let Some(v) = v {
                    self.p.out.push(' ');
                    self.expr(v);
                }
                self.p.out.push_str(")\n");
            }
            Stmt::Empty => self.p.out.push_str("empty\n"),
        }
    }

    fn list(&mut self, r: Range) {
        let u = self.p.u;
        for &a in u.args(r) {
            self.p.out.push(' ');
            self.expr(a);
        }
        self.p.out.push(')');
    }

    fn expr(&mut self, e: ExprId) {
        let u = self.p.u;
        match *u.expr(e) {
            Expr::Int(v, suffix) => {
                let _ = write!(self.p.out, "{v}:{suffix:?}");
            }
            // the bits: -0.0 and 0.0, and every NaN, stay apart
            Expr::Float(v, suffix) => {
                let _ = write!(self.p.out, "{v:?}#{:016x}:{suffix:?}", v.to_bits());
            }
            Expr::Bool(b) => {
                let _ = write!(self.p.out, "{b}");
            }
            Expr::Ident(s) => self.p.name(s),
            Expr::Unary(op, a) => {
                let _ = write!(self.p.out, "({op:?} ");
                self.expr(a);
                self.p.out.push(')');
            }
            Expr::Binary(op, a, b) => {
                let _ = write!(self.p.out, "({op:?} ");
                self.expr(a);
                self.p.out.push(' ');
                self.expr(b);
                self.p.out.push(')');
            }
            Expr::Assign(op, a, b) => {
                match op {
                    Some(op) => {
                        let _ = write!(self.p.out, "(Assign{op:?} ");
                    }
                    None => self.p.out.push_str("(Assign "),
                }
                self.expr(a);
                self.p.out.push(' ');
                self.expr(b);
                self.p.out.push(')');
            }
            Expr::Ternary(c, t, f) => {
                self.p.out.push_str("(? ");
                self.expr(c);
                self.p.out.push(' ');
                self.expr(t);
                self.p.out.push(' ');
                self.expr(f);
                self.p.out.push(')');
            }
            Expr::Call(name, args) => {
                let _ = write!(self.p.out, "(call {}", u.name(name));
                self.list(args);
            }
            Expr::Construct(t, args) => {
                self.p.out.push_str("(new ");
                self.p.ty(t);
                self.list(args);
            }
            Expr::Cast(t, a) => {
                self.p.out.push_str("(cast ");
                self.p.ty(t);
                self.p.out.push(' ');
                self.expr(a);
                self.p.out.push(')');
            }
            Expr::Method(name, all) => {
                let (o, args) = u.method(all);
                self.p.out.push_str("(method ");
                self.expr(o);
                let _ = write!(self.p.out, " {}", u.name(name));
                self.list(args);
            }
            Expr::Member(o, name) => {
                self.p.out.push_str("(. ");
                self.expr(o);
                let _ = write!(self.p.out, " {})", u.name(name));
            }
            Expr::Index(o, i) => {
                self.p.out.push_str("([] ");
                self.expr(o);
                self.p.out.push(' ');
                self.expr(i);
                self.p.out.push(')');
            }
            Expr::InitList(r) => {
                self.p.out.push_str("(init");
                self.list(r);
            }
        }
    }
}

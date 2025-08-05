use crate::lex::{Lexer, Token};

#[derive(Debug, Clone, PartialEq)]
pub enum BaseType {
    I8,
    I32,
    I64,
    F32,
    F64,
    Ptr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Base(BaseType),
    Array(BaseType, u64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Register(String),
    Global(String),
    Int(i64),
    Float(f64),
    Number(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Udiv,
    Rem,
    Urem,
    And,
    Or,
    Xor,
    Lsl,
    Lsr,
    Asr,
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConvOp {
    Sext,
    Zext,
    Trunc,
    Itof,
    Uitof,
    Ftoi,
    Fpromote,
    Fdemote,
    Ptoi,
    Itop,
    Bitcast,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    Arith {
        dest: String,
        op: ArithOp,
        ty: BaseType,
        operands: Vec<Value>,
    },
    Memory {
        dest: Option<String>,
        op: MemOp,
        ty: BaseType,
        operands: Vec<Value>,
    },
    Compare {
        dest: String,
        op: CmpOp,
        ty: BaseType,
        left: Value,
        right: Value,
    },
    Select {
        dest: String,
        ty: BaseType,
        condition: Value,
        true_val: Value,
        false_val: Value,
    },
    Convert {
        dest: String,
        op: ConvOp,
        dest_ty: BaseType,
        operand: Value,
    },
    Call {
        dest: Option<String>,
        function: String,
        args: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemOp {
    Load,
    Store,
    Alloc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchTarget {
    pub block: String,
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Branch(BranchTarget),
    BranchIf(Value, BranchTarget, BranchTarget),
    Return(Option<Value>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: String,
    pub params: Vec<(String, BaseType)>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, BaseType)>,
    pub return_type: Option<BaseType>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunction {
    pub name: String,
    pub param_types: Vec<BaseType>,
    pub return_type: Option<BaseType>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataDef {
    pub name: String,
    pub ty: Type,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Definition {
    Data(DataDef),
    Function(Function),
    ExternFunction(ExternFunction),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub definitions: Vec<Definition>,
}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(mut lexer: Lexer) -> Result<Self, String> {
        let tokens = lexer.tokenize()?;
        Ok(Parser {
            tokens,
            position: 0,
        })
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::Lexer;

    #[test]
    fn test_parse_hello_world() {
        let input = r#"
data @hello_world_z: [i8; 14] = "Hello, World\00"

declare fn @puts(ptr) -> i32

fn @main() -> i32 {
start:
    %r = call @puts(@hello_world_z)
    ret %r
}
"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        assert_eq!(program.definitions.len(), 3);

        if let Definition::Data(data) = &program.definitions[0] {
            assert_eq!(data.name, "hello_world_z");
            assert_eq!(data.value, "Hello, World\0");
        } else {
            panic!("Expected data definition");
        }

        if let Definition::ExternFunction(func) = &program.definitions[1] {
            assert_eq!(func.name, "puts");
            assert_eq!(func.param_types, vec![BaseType::Ptr]);
            assert_eq!(func.return_type, Some(BaseType::I32));
        } else {
            panic!("Expected extern function");
        }

        if let Definition::Function(func) = &program.definitions[2] {
            assert_eq!(func.name, "main");
            assert_eq!(func.return_type, Some(BaseType::I32));
            assert_eq!(func.blocks.len(), 1);

            let block = &func.blocks[0];
            assert_eq!(block.name, "start");
            assert_eq!(block.instructions.len(), 1);

            if let Instruction::Call {
                dest,
                function,
                args,
            } = &block.instructions[0]
            {
                assert_eq!(dest, &Some("r".to_string()));
                assert_eq!(function, "puts");
                assert_eq!(args.len(), 1);
            } else {
                panic!("Expected call instruction");
            }
        } else {
            panic!("Expected function definition");
        }
    }

    #[test]
    fn test_parse_arithmetic() {
        let input = r#"
fn @test() -> i32 {
start:
    %result = add.i32 %a, %b
    ret %result
}
"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];
            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[0]
            {
                assert_eq!(dest, "result");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands.len(), 2);
            } else {
                panic!("Expected arithmetic instruction");
            }
        }
    }

    #[test]
    fn test_two_functions() {
        let input = r#"
fn @add(%a: i32, %b: i32) -> i32 {
entry(%a: i32, %b: i32):
    %sum = add.i32 %a, %b
    ret %sum
}

fn @main() -> i32 {
start:
    %val_a = add.i32 10, 20
    %val_b = add.i32 5, 7
    %total = call @add(%val_a, %val_b)
    ret %total
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        assert_eq!(program.definitions.len(), 2);

        if let Definition::Function(add_func) = &program.definitions[0] {
            assert_eq!(add_func.name, "add");
            assert_eq!(add_func.params.len(), 2);
            assert_eq!(add_func.params[0], ("a".to_string(), BaseType::I32));
            assert_eq!(add_func.params[1], ("b".to_string(), BaseType::I32));
            assert_eq!(add_func.return_type, Some(BaseType::I32));
            assert_eq!(add_func.blocks.len(), 1);

            let block = &add_func.blocks[0];
            assert_eq!(block.name, "entry");
            assert_eq!(block.params[0], ("a".to_string(), BaseType::I32));
            assert_eq!(block.params[1], ("b".to_string(), BaseType::I32));
            assert_eq!(block.instructions.len(), 1);

            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[0]
            {
                assert_eq!(dest, "sum");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[0], Value::Register("a".to_string()));
                assert_eq!(operands[1], Value::Register("b".to_string()));
            } else {
                panic!("Expected arithmetic instruction in @add");
            }

            if let Terminator::Return(Some(val)) = &block.terminator {
                assert_eq!(*val, Value::Register("sum".to_string()));
            } else {
                panic!("Expected return terminator in @add");
            }
        } else {
            panic!("Expected @add function as first definition");
        }

        if let Definition::Function(main_func) = &program.definitions[1] {
            assert_eq!(main_func.name, "main");
            assert!(main_func.params.is_empty());
            assert_eq!(main_func.return_type, Some(BaseType::I32));
            assert_eq!(main_func.blocks.len(), 1);

            let block = &main_func.blocks[0];
            assert_eq!(block.name, "start");
            assert_eq!(block.instructions.len(), 3);

            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[0]
            {
                assert_eq!(dest, "val_a");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands, &vec![Value::Number(10), Value::Number(20)]);
            } else {
                panic!("Expected add.i32 instruction for val_a");
            }

            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[1]
            {
                assert_eq!(dest, "val_b");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands, &vec![Value::Number(5), Value::Number(7)]);
            } else {
                panic!("Expected add.i32 instruction for val_b");
            }

            if let Instruction::Call {
                dest,
                function,
                args,
            } = &block.instructions[2]
            {
                assert_eq!(dest, &Some("total".to_string()));
                assert_eq!(function, "add");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Value::Register("val_a".to_string()));
                assert_eq!(args[1], Value::Register("val_b".to_string()));
            } else {
                panic!("Expected call instruction for @add");
            }

            if let Terminator::Return(Some(val)) = &block.terminator {
                assert_eq!(*val, Value::Register("total".to_string()));
            } else {
                panic!("Expected return terminator in @main");
            }
        } else {
            panic!("Expected @main function as second definition");
        }
    }

    #[test]
    fn test_block_without_terminator() {
        let input = r#"
fn @test() -> i32 {
start:
    %a = add.i32 10, 20
    %b = add.i32 5, 7
    %total = call @add(%a, %b)
}"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let poss_err = parser.parse();
        assert!(poss_err.is_err()); // TODO: improve this error message, as we aren't getting
        // 'Expected terminator', but rather we're erroring on the '}'
    }
}

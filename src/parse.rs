use crate::lex::{Lexer, Token};
use anyhow::{Result, anyhow};

trait FromToken {
    fn from_token(token: &Token) -> Result<Self>
    where
        Self: Sized;
}

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

impl Type {
    fn into_base_type(self) -> BaseType {
        match self {
            Type::Base(base_ty) => base_ty,
            _ => panic!("Expected base type, got {:?}", self),
        }
    }
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

impl FromToken for ArithOp {
    fn from_token(token: &Token) -> Result<Self> {
        match token {
            Token::Add => Ok(ArithOp::Add),
            Token::Sub => Ok(ArithOp::Sub),
            Token::Mul => Ok(ArithOp::Mul),
            Token::Div => Ok(ArithOp::Div),
            Token::Udiv => Ok(ArithOp::Udiv),
            Token::Rem => Ok(ArithOp::Rem),
            Token::Urem => Ok(ArithOp::Urem),
            Token::And => Ok(ArithOp::And),
            Token::Or => Ok(ArithOp::Or),
            Token::Xor => Ok(ArithOp::Xor),
            Token::Lsl => Ok(ArithOp::Lsl),
            Token::Lsr => Ok(ArithOp::Lsr),
            Token::Asr => Ok(ArithOp::Asr),
            Token::Neg => Ok(ArithOp::Neg),
            _ => Err(anyhow!("Expected arithmetic operation, got {:?}", token)),
        }
    }
}

impl ArithOp {
    fn num_operands(&self) -> usize {
        match self {
            ArithOp::Add
            | ArithOp::Sub
            | ArithOp::Mul
            | ArithOp::Div
            | ArithOp::Udiv
            | ArithOp::Rem
            | ArithOp::Urem
            | ArithOp::And
            | ArithOp::Or
            | ArithOp::Xor
            | ArithOp::Lsl
            | ArithOp::Lsr
            | ArithOp::Asr => 2,
            ArithOp::Neg => 1,
        }
    }
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

impl FromToken for CmpOp {
    fn from_token(token: &Token) -> Result<Self> {
        match token {
            Token::Eq => Ok(CmpOp::Eq),
            Token::Ne => Ok(CmpOp::Ne),
            Token::Lt => Ok(CmpOp::Lt),
            Token::Le => Ok(CmpOp::Le),
            Token::Gt => Ok(CmpOp::Gt),
            Token::Ge => Ok(CmpOp::Ge),
            Token::Ult => Ok(CmpOp::Ult),
            Token::Ule => Ok(CmpOp::Ule),
            Token::Ugt => Ok(CmpOp::Ugt),
            Token::Uge => Ok(CmpOp::Uge),
            _ => Err(anyhow!("Expected comparison operation, got {:?}", token)),
        }
    }
}

impl CmpOp {
    fn num_operands(&self) -> usize {
        2
    }
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

impl FromToken for ConvOp {
    fn from_token(token: &Token) -> Result<Self> {
        match token {
            Token::Sext => Ok(ConvOp::Sext),
            Token::Zext => Ok(ConvOp::Zext),
            Token::Trunc => Ok(ConvOp::Trunc),
            Token::Itof => Ok(ConvOp::Itof),
            Token::Uitof => Ok(ConvOp::Uitof),
            Token::Ftoi => Ok(ConvOp::Ftoi),
            Token::Fpromote => Ok(ConvOp::Fpromote),
            Token::Fdemote => Ok(ConvOp::Fdemote),
            Token::Ptoi => Ok(ConvOp::Ptoi),
            Token::Itop => Ok(ConvOp::Itop),
            Token::Bitcast => Ok(ConvOp::Bitcast),
            _ => Err(anyhow!("Expected conversion operation, got {:?}", token)),
        }
    }
}

impl ConvOp {
    fn num_operands(&self) -> usize {
        1
    }
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

impl FromToken for MemOp {
    fn from_token(token: &Token) -> Result<Self> {
        match token {
            Token::Load => Ok(MemOp::Load),
            Token::Store => Err(anyhow!("Stores should not be created from_token")),
            Token::Alloc => Ok(MemOp::Alloc),
            _ => Err(anyhow!("Expected memory operation, got {:?}", token)),
        }
    }
}

impl MemOp {
    fn num_operands(&self) -> usize {
        match self {
            MemOp::Load | MemOp::Store => 2,
            MemOp::Alloc => 1,
        }
    }
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
    pub fn new(mut lexer: Lexer) -> Result<Self> {
        let tokens = lexer.tokenize()?;
        println!("{:?}", tokens);
        Ok(Parser {
            tokens,
            position: 0,
        })
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.position).cloned();
        self.position += 1;
        tok
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn expect_token(&mut self, expected: Token) -> Result<()> {
        let actual = self.peek().ok_or(anyhow!("Unexpected end of file"))?;
        if actual == &expected {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!("Expected {:?}, got {:?}", expected, actual))
        }
    }

    fn expect_global(&mut self) -> Result<String> {
        let actual = self.peek().ok_or(anyhow!("Unexpected end of file"))?;
        if let Token::Global(ident) = actual {
            let ident = ident.clone();
            self.advance();
            Ok(ident)
        } else {
            Err(anyhow!("Expected global, got {:?}", actual))
        }
    }

    fn type_annotation(&mut self) -> Result<Option<BaseType>> {
        if let Token::Colon = self.next().ok_or(anyhow!("Unexpected end of file"))? {
            let ty = self.type_decl()?;
            Ok(Some(ty.into_base_type()))
        } else {
            Ok(None)
        }
    }

    fn operands(&mut self, num_operands: usize) -> Result<Vec<Value>> {
        let mut operands = Vec::new();
        for _ in 0..num_operands {
            let tok = self.next().ok_or(anyhow!("Unexpected end of file"))?;
            if let Token::Register(reg) = tok {
                operands.push(Value::Register(reg.clone()));
            } else if let Token::Int(num) = tok {
                operands.push(Value::Int(num));
            } else {
                return Err(anyhow!("Expected register or number, got {:?}", tok));
            }
        }

        Ok(operands)
    }

    fn definition(&mut self) -> Result<Definition> {
        let definition = match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            Token::Data => Definition::Data(self.data_def()?),
            Token::Fn => Definition::Function(self.fn_def()?),
            // Token::Extern => self.extern_function()?.into(),
            _ => {
                return Err(anyhow!(
                    "Expected definition, got {:?}, at position {}",
                    self.peek(),
                    self.position
                ));
            }
        };

        Ok(definition)
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut definitions = Vec::new();
        while let Some(tok) = self.peek() {
            if tok == &Token::Eof {
                break;
            }
            let definition = self.definition()?;
            definitions.push(definition);
        }

        Ok(Program { definitions })
    }
}

// Defintion specific parsing
impl Parser {
    fn type_decl(&mut self) -> Result<Type> {
        let ty = match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            Token::I8 => Type::Base(BaseType::I8),
            Token::I32 => Type::Base(BaseType::I32),
            Token::I64 => Type::Base(BaseType::I64),
            Token::F32 => Type::Base(BaseType::F32),
            Token::F64 => Type::Base(BaseType::F64),
            Token::Ptr => Type::Base(BaseType::Ptr),
            Token::LeftBracket => {
                self.advance();
                let inner_ty = self.type_decl()?;
                let inner_base_ty = match inner_ty {
                    Type::Base(base_ty) => base_ty,
                    _ => return Err(anyhow!("Expected base type, got {:?}", inner_ty)),
                };
                self.expect_token(Token::Semicolon)?;
                let Token::Number(size) = self.peek().ok_or(anyhow!("Unexpected end of file"))?
                else {
                    return Err(anyhow!("Expected number, got {:?}", self.peek()));
                };
                let size = *size; // required to drop the immutable borrow
                self.advance();
                self.expect_token(Token::RightBracket)?;
                // we return here cause the advance call at the bottom will advance too far
                return Ok(Type::Array(inner_base_ty, size));
            }
            _ => return Err(anyhow!("Expected type, got {:?}", self.peek())),
        };
        self.advance();
        Ok(ty)
    }

    fn data_def(&mut self) -> Result<DataDef> {
        self.expect_token(Token::Data)?;
        let name = self.expect_global()?;
        self.expect_token(Token::Colon)?;
        let ty = self.type_decl()?;
        self.expect_token(Token::Equals)?;
        let Token::String(value) = self.peek().ok_or(anyhow!("Unexpected end of file"))? else {
            return Err(anyhow!("Expected string, got {:?}", self.peek()));
        };
        let value = value.clone();
        self.advance();
        Ok(DataDef { name, ty, value })
    }

    fn block_fn_param(&mut self) -> Result<Vec<(String, BaseType)>> {
        let mut params = Vec::new();
        if let Token::LeftParen = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            self.advance();
            while let Token::Register(reg) = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
                let reg = reg.clone();
                self.advance();
                self.expect_token(Token::Colon)?;
                let ty = self.type_decl()?;
                let ty = ty.into_base_type();
                params.push((reg, ty));
                if let Token::Comma = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
                    self.advance();
                } else {
                    break;
                };
            }
            match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
                Token::RightParen => {
                    self.advance();
                }
                _ => return Err(anyhow!("Expected right paren, got {:?}", self.peek())),
            }
        }
        Ok(params)
    }

    fn fn_def(&mut self) -> Result<Function> {
        self.expect_token(Token::Fn)?;
        let name = self.expect_global()?;
        let params =
            if let Token::LeftParen = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
                self.block_fn_param()?
            } else {
                panic!("Expected left paren, got {:?}", self.peek());
            };

        let return_type =
            if let Token::Arrow = self.next().ok_or(anyhow!("Unexpected end of file"))? {
                let return_type = self.type_decl()?;
                Some(return_type.into_base_type())
            } else {
                None
            };
        self.expect_token(Token::LeftBrace);
        let blocks = self.blocks(&return_type)?;
        self.expect_token(Token::RightBrace);
        Ok(Function {
            name,
            params,
            return_type,
            blocks,
        })
    }

    fn blocks(&mut self, return_type: &Option<BaseType>) -> Result<Vec<Block>> {
        let mut blocks = Vec::new();
        while !matches!(self.peek(), Some(Token::RightBrace)) {
            blocks.push(self.block(return_type)?);
        }
        Ok(blocks)
    }

    fn block(&mut self, return_type: &Option<BaseType>) -> Result<Block> {
        let name =
            if let Token::Ident(ident) = self.next().ok_or(anyhow!("Unexpected end of file"))? {
                ident
            } else {
                panic!("Expected block name, got {:?}", self.peek());
            };

        let params = self.block_fn_param()?;
        self.expect_token(Token::Colon);
        let instructions = self.insts()?;
        let terminator = self.terminator(return_type)?;

        Ok(Block {
            name,
            params,
            instructions,
            terminator,
        })
    }

    fn insts(&mut self) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        loop {
            if let Ok(inst) = self.inst() {
                instructions.push(inst);
            } else {
                break;
            }
        }

        Ok(instructions)
    }

    fn inst(&mut self) -> Result<Instruction> {
        match self.next().ok_or(anyhow!("Unexpected end of file"))? {
            Token::Register(reg) => {
                let Token::Equals = self.next().ok_or(anyhow!("Unexpected end of file"))? else {
                    return Err(anyhow!(
                        "Expected '=' after register, got {:?}",
                        self.peek()
                    ));
                };

                let tok_peek = self.peek().ok_or(anyhow!("Unexpected end of file"))?;
                if let Ok(op) = ArithOp::from_token(tok_peek) {
                    let ty = self.type_annotation()?;
                    let operands = self.operands(op.num_operands())?;

                    #[deny(clippy::implicit_return)]
                    return Ok(Instruction::Arith {
                        dest: reg,
                        op,
                        ty: ty.unwrap_or(BaseType::I32),
                        operands,
                    });
                } else if let Ok(op) = MemOp::from_token(tok_peek) {
                    let ty = self.type_annotation()?;
                    let operands = self.operands(op.num_operands())?;

                    return Ok(Instruction::Memory {
                        dest: Some(reg),
                        op,
                        ty: ty.unwrap_or(BaseType::I32),
                        operands,
                    });
                } else if let Ok(op) = CmpOp::from_token(tok_peek) {
                    let ty = self.type_annotation()?;
                    let left = self.operands(1)?.pop().unwrap();
                    let right = self.operands(1)?.pop().unwrap();

                    return Ok(Instruction::Compare {
                        dest: reg,
                        op,
                        ty: ty.unwrap_or(BaseType::I32),
                        left,
                        right,
                    });
                } else if let Ok(op) = ConvOp::from_token(tok_peek) {
                    let ty = self.type_annotation()?;
                    let operand = self.operands(1)?.pop().unwrap();

                    return Ok(Instruction::Convert {
                        dest: reg,
                        op,
                        dest_ty: ty.unwrap_or(BaseType::I32),
                        operand,
                    });
                } else {
                    return Err(anyhow!(
                        "Expected arithmetic, memory, compare, or convert operation, got {:?}",
                        tok_peek
                    ));
                }
            }
            _ => todo!(),
        }
    }

    fn terminator(&mut self, ret_type: &Option<BaseType>) -> Result<Terminator> {
        match self.next().ok_or(anyhow!("Unexpected end of file"))? {
            Token::Ret => {
                if ret_type.is_some() {
                    let val = self.operands(1)?.pop().unwrap();
                    Ok(Terminator::Return(Some(val)))
                } else {
                    Ok(Terminator::Return(None))
                }
            }
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::Lexer;

    #[test]
    fn test_parse_data() {
        let input = r#"
data @hello_world_z: [i8; 14] = "Hello, World\00"
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).expect("Failed to create parser");
        let result = parser.parse();
        println!("{:?}", result);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert_eq!(module.definitions.len(), 1);
        let data = &module.definitions[0];
        let Definition::Data(data) = data else {
            panic!("Expected data definition");
        };
        assert_eq!(data.name, "hello_world_z");
        assert_eq!(data.value, "Hello, World\0");
    }

    #[test]
    fn test_parse_empty_fn() {
        let input = r#"
fn @empty() {}

fn @main() -> i32 {}
"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).expect("Failed to create parser");
        let result = parser.parse();
        println!("{:?}", result);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert_eq!(module.definitions.len(), 2);

        if let Definition::Function(func) = &module.definitions[0] {
            assert_eq!(func.name, "empty");
            assert_eq!(func.params.len(), 0);
            assert_eq!(func.return_type, None);
            assert_eq!(func.blocks.len(), 0);
        } else {
            panic!("Expected function definition");
        }

        if let Definition::Function(func) = &module.definitions[1] {
            assert_eq!(func.name, "main");
            assert_eq!(func.params.len(), 0);
            assert_eq!(func.return_type, Some(BaseType::I32));
            assert_eq!(func.blocks.len(), 0);
        } else {
            panic!("Expected function definition");
        }
    }

    #[test]
    fn test_parse_fn_with_entry_block() {
        let input = r#"
fn @add(%a: i32, %b: i32) -> i32 {
entry(%a: i32, %b: i32):
    %sum = add.i32 %a, %b
    ret %sum
}
"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).expect("Failed to create parser");
        let result = parser.parse();
        println!("{:?}", result);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert_eq!(module.definitions.len(), 1);

        if let Definition::Function(func) = &module.definitions[0] {
            assert_eq!(func.name, "add");
            assert_eq!(func.params.len(), 2);
            assert_eq!(func.params[0], ("a".to_string(), BaseType::I32));
            assert_eq!(func.params[1], ("b".to_string(), BaseType::I32));
            assert_eq!(func.return_type, Some(BaseType::I32));
            assert_eq!(func.blocks.len(), 1);

            let block = &func.blocks[0];
            assert_eq!(block.name, "entry");
            assert_eq!(block.params.len(), 2);
            assert_eq!(block.params[0], ("a".to_string(), BaseType::I32));
            assert_eq!(block.params[1], ("b".to_string(), BaseType::I32));
            assert_eq!(block.instructions.len(), 0);
        } else {
            panic!("Expected function definition");
        }
    }

    #[ignore]
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

    #[ignore]
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

    #[ignore]
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

    #[ignore]
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

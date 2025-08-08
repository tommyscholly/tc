use std::collections::HashMap;

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
    // Number(u64),
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
pub enum GlobalData {
    Str(String),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataDef {
    pub name: String,
    pub ty: Type,
    pub value: GlobalData,
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

    fn is_instruction_token(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Register(_)) | Some(Token::Call) | Some(Token::Store)
        )
    }

    fn is_arithmetic_op(&self, token: &Token) -> bool {
        matches!(
            token,
            Token::Add
                | Token::Sub
                | Token::Mul
                | Token::Div
                | Token::Udiv
                | Token::Rem
                | Token::Urem
                | Token::And
                | Token::Or
                | Token::Xor
                | Token::Lsl
                | Token::Lsr
                | Token::Asr
                | Token::Neg
        )
    }

    fn is_memory_op(&self, token: &Token) -> bool {
        matches!(token, Token::Load | Token::Alloc)
    }

    fn is_compare_op(&self, token: &Token) -> bool {
        matches!(
            token,
            Token::Eq
                | Token::Ne
                | Token::Lt
                | Token::Le
                | Token::Gt
                | Token::Ge
                | Token::Ult
                | Token::Ule
                | Token::Ugt
                | Token::Uge
        )
    }

    fn is_convert_op(&self, token: &Token) -> bool {
        matches!(
            token,
            Token::Sext
                | Token::Zext
                | Token::Trunc
                | Token::Itof
                | Token::Uitof
                | Token::Ftoi
                | Token::Fpromote
                | Token::Fdemote
                | Token::Ptoi
                | Token::Itop
                | Token::Bitcast
        )
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

    fn expect_ident(&mut self) -> Result<String> {
        let actual = self.peek().ok_or(anyhow!("Unexpected end of file"))?;
        if let Token::Ident(ident) = actual {
            let ident = ident.clone();
            self.advance();
            Ok(ident)
        } else {
            Err(anyhow!("Expected identifier, got {:?}", actual))
        }
    }

    fn type_annotation(&mut self) -> Result<Option<BaseType>> {
        if let Token::Dot = self.next().ok_or(anyhow!("Unexpected end of file"))? {
            let ty = self.type_decl()?;
            Ok(Some(ty.into_base_type()))
        } else {
            Ok(None)
        }
    }

    fn operands(&mut self, num_operands: usize) -> Result<Vec<Value>> {
        let mut operands = Vec::new();
        for i in 0..num_operands {
            let tok = self.next().ok_or(anyhow!("Unexpected end of file"))?;
            if let Token::Register(reg) = tok {
                operands.push(Value::Register(reg.clone()));
            } else if let Token::Int(num) = tok {
                operands.push(Value::Int(num));
            } else if let Token::Global(global) = tok {
                operands.push(Value::Global(global.clone()));
            } else {
                return Err(anyhow!("Expected register or number, got {:?}", tok));
            }

            if i + 1 < num_operands {
                self.expect_token(Token::Comma)?;
            }
        }

        Ok(operands)
    }

    fn definition(&mut self) -> Result<Definition> {
        let definition = match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            Token::Data => Definition::Data(self.data_def()?),
            Token::Fn => Definition::Function(self.fn_def()?),
            Token::Declare => Definition::ExternFunction(self.extern_function()?),
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
                let Token::Int(size) = self.peek().ok_or(anyhow!("Unexpected end of file"))? else {
                    return Err(anyhow!("Expected number, got {:?}", self.peek()));
                };
                assert!(*size >= 0);
                let size = *size; // required to drop the immutable borrow
                self.advance();
                self.expect_token(Token::RightBracket)?;
                // we return here cause the advance call at the bottom will advance too far
                // @SAFETY: we checked that size is >= 0
                return Ok(Type::Array(inner_base_ty, size as u64));
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
        match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            Token::String(value) => {
                let value = value.clone();
                self.advance();
                Ok(DataDef {
                    name,
                    ty,
                    value: GlobalData::Str(value),
                })
            }
            Token::Int(value) => {
                let value = *value;
                self.advance();
                Ok(DataDef {
                    name,
                    ty,
                    value: GlobalData::Int(value),
                })
            }
            Token::Float(value) => {
                let value = *value;
                self.advance();
                Ok(DataDef {
                    name,
                    ty,
                    value: GlobalData::Float(value),
                })
            }
            _ => Err(anyhow!("Expected string, got {:?}", self.peek())),
        }
    }

    fn type_params(&mut self) -> Result<Vec<BaseType>> {
        let mut params = Vec::new();
        if let Token::LeftParen = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            self.advance();
            while let Ok(ty) = self.type_decl() {
                params.push(ty.into_base_type());
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

    fn extern_function(&mut self) -> Result<ExternFunction> {
        self.expect_token(Token::Declare)?;
        self.expect_token(Token::Fn)?;
        let name = self.expect_global()?;
        let param_types = self.type_params()?;

        let return_type =
            if let Token::Arrow = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
                self.expect_token(Token::Arrow)?;
                Some(self.type_decl()?.into_base_type())
            } else {
                None
            };

        Ok(ExternFunction {
            name,
            param_types,
            return_type,
        })
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

        let fn_ = Function {
            name,
            params,
            return_type,
            blocks,
        };
        println!("{:?}", fn_);
        Ok(fn_)
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
        println!("\n insts: {:?}", instructions);
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

        while self.is_instruction_token() {
            instructions.push(self.inst()?);
        }

        Ok(instructions)
    }

    fn call_operands(&mut self) -> Result<Vec<Value>> {
        self.expect_token(Token::LeftParen)?;

        let mut call_ops = Vec::new();
        while !matches!(self.peek(), Some(Token::RightParen)) {
            let value = self.operands(1)?.pop().unwrap();
            println!("{:?}", value);
            call_ops.push(value);
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else if let Token::RightParen =
                self.peek().ok_or(anyhow!("Unexpected end of file"))?
            {
                break;
            } else {
                return Err(anyhow!("Expected ',' or ')', got {:?}", self.peek()));
            }
        }
        self.advance();

        Ok(call_ops)
    }

    fn call(&mut self, dest: Option<String>) -> Result<Instruction> {
        self.expect_token(Token::Call)?;
        let function = self.expect_global()?;
        let call_ops = self.call_operands()?;

        Ok(Instruction::Call {
            dest,
            function: function.clone(),
            args: call_ops,
        })
    }

    fn inst(&mut self) -> Result<Instruction> {
        match self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            Token::Register(reg) => {
                let reg = reg.clone();
                self.advance();
                let Token::Equals = self.next().ok_or(anyhow!("Unexpected end of file"))? else {
                    return Err(anyhow!(
                        "Expected '=' after register, got {:?}",
                        self.peek()
                    ));
                };

                self.parse_operation_instruction(reg)
            }
            Token::Store => {
                println!("Store");
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annotation");
                let operands = self.operands(2)?;
                Ok(Instruction::Memory {
                    dest: None,
                    op: MemOp::Store,
                    ty,
                    operands,
                })
            }
            Token::Call => self.call(None),
            _ => Err(anyhow!("Expected instruction token, got {:?}", self.peek())),
        }
    }

    fn parse_operation_instruction(&mut self, dest: String) -> Result<Instruction> {
        let token = self.peek().ok_or(anyhow!("Unexpected end of file"))?;

        match token {
            token if self.is_arithmetic_op(token) => {
                let op = ArithOp::from_token(token)?;
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annontation");
                let operands = self.operands(op.num_operands())?;
                Ok(Instruction::Arith {
                    dest,
                    op,
                    ty,
                    operands,
                })
            }
            token if self.is_memory_op(token) => {
                let op = MemOp::from_token(token)?;
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annontation");
                let operands = self.operands(op.num_operands())?;
                Ok(Instruction::Memory {
                    dest: Some(dest),
                    op,
                    ty,
                    operands,
                })
            }
            token if self.is_compare_op(token) => {
                let op = CmpOp::from_token(token)?;
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annontation");
                let mut operands = self.operands(2)?;
                let right = operands.pop().unwrap();
                let left = operands.pop().unwrap();
                Ok(Instruction::Compare {
                    dest,
                    op,
                    ty,
                    left,
                    right,
                })
            }
            token if self.is_convert_op(token) => {
                let op = ConvOp::from_token(token)?;
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annontation");
                let operand = self.operands(1)?.pop().unwrap();
                Ok(Instruction::Convert {
                    dest,
                    op,
                    dest_ty: ty,
                    operand,
                })
            }
            Token::Select => {
                self.advance();
                let ty = self.type_annotation()?.expect("Expected type annotation");
                let mut cond_true_false = self.operands(3)?;
                let false_val = cond_true_false.pop().unwrap();
                let true_val = cond_true_false.pop().unwrap();
                let condition = cond_true_false.pop().unwrap();

                Ok(Instruction::Select {
                    dest,
                    ty,
                    condition,
                    true_val,
                    false_val,
                })
            }
            Token::Call => self.call(Some(dest)),

            _ => Err(anyhow!("Expected operation, got {:?}", token)),
        }
    }

    fn branch_target(&mut self) -> Result<BranchTarget> {
        let block = self.expect_ident()?;
        let args = if let Token::LeftParen = self.peek().ok_or(anyhow!("Unexpected end of file"))? {
            self.call_operands()?
        } else {
            Vec::new()
        };
        Ok(BranchTarget { block, args })
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
            Token::Br => {
                let target = self.branch_target()?;
                Ok(Terminator::Branch(target))
            }
            Token::Brif => {
                let condition = self.operands(1)?.pop().unwrap();
                self.expect_token(Token::Comma)?;
                let true_target = self.branch_target()?;
                self.expect_token(Token::Comma)?;
                let false_target = self.branch_target()?;
                Ok(Terminator::BranchIf(condition, true_target, false_target))
            }
            t => Err(anyhow!("Expected terminator, got {:?}", t)),
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
        assert_eq!(data.value, GlobalData::Str("Hello, World\0".to_string()));
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
            assert_eq!(
                block.instructions[0],
                Instruction::Arith {
                    dest: "sum".to_string(),
                    op: ArithOp::Add,
                    ty: BaseType::I32,
                    operands: vec![
                        Value::Register("a".to_string()),
                        Value::Register("b".to_string())
                    ]
                }
            );
            assert_eq!(
                block.terminator,
                Terminator::Return(Some(Value::Register("sum".to_string())))
            );
        } else {
            panic!("Expected function definition");
        }
    }

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
            assert_eq!(data.value, GlobalData::Str("Hello, World\0".to_string()));
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
                assert_eq!(operands, &vec![Value::Int(10), Value::Int(20)]);
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
                assert_eq!(operands, &vec![Value::Int(5), Value::Int(7)]);
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

    #[test]
    fn test_multi_block_fn() {
        let input = r#"
fn @main() -> i32 {
start:
    %a = add.i32 0, 1
    br loop(%a)
loop(%i: i32):
    %b = add.i32 %i, 1
    %cond = eq.i32 %i, 10
    brif %cond, end(%b), loop(%b)
end(%i: i32):
    ret %i
}
"#;

        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        assert_eq!(program.definitions.len(), 1);

        if let Definition::Function(func) = &program.definitions[0] {
            assert_eq!(func.name, "main");
            assert!(func.params.is_empty());
            assert_eq!(func.return_type, Some(BaseType::I32));
            assert_eq!(func.blocks.len(), 3);

            let start_block = &func.blocks[0];
            assert_eq!(start_block.name, "start");
            assert!(start_block.params.is_empty());
            assert_eq!(start_block.instructions.len(), 1);

            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &start_block.instructions[0]
            {
                assert_eq!(dest, "a");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands, &vec![Value::Int(0), Value::Int(1)]);
            } else {
                panic!("Expected arithmetic instruction in start block");
            }

            if let Terminator::Branch(target) = &start_block.terminator {
                assert_eq!(target.block, "loop");
                assert_eq!(target.args.len(), 1);
                assert_eq!(target.args[0], Value::Register("a".to_string()));
            } else {
                panic!("Expected branch terminator in start block");
            }

            let loop_block = &func.blocks[1];
            assert_eq!(loop_block.name, "loop");
            assert_eq!(loop_block.params.len(), 1);
            assert_eq!(loop_block.params[0], ("i".to_string(), BaseType::I32));
            assert_eq!(loop_block.instructions.len(), 2);

            if let Instruction::Arith {
                dest,
                op,
                ty,
                operands,
            } = &loop_block.instructions[0]
            {
                assert_eq!(dest, "b");
                assert_eq!(*op, ArithOp::Add);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(
                    operands,
                    &vec![Value::Register("i".to_string()), Value::Int(1)]
                );
            } else {
                panic!("Expected arithmetic instruction for %b in loop block");
            }

            if let Instruction::Compare {
                dest,
                op,
                ty,
                left,
                right,
            } = &loop_block.instructions[1]
            {
                assert_eq!(dest, "cond");
                assert_eq!(*op, CmpOp::Eq);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(*left, Value::Register("i".to_string()));
                assert_eq!(*right, Value::Int(10));
            } else {
                panic!("Expected compare instruction for %cond in loop block");
            }

            if let Terminator::BranchIf(condition, true_target, false_target) =
                &loop_block.terminator
            {
                assert_eq!(*condition, Value::Register("cond".to_string()));

                assert_eq!(true_target.block, "end");
                assert_eq!(true_target.args.len(), 1);
                assert_eq!(true_target.args[0], Value::Register("b".to_string()));

                assert_eq!(false_target.block, "loop");
                assert_eq!(false_target.args.len(), 1);
                assert_eq!(false_target.args[0], Value::Register("b".to_string()));
            } else {
                panic!("Expected conditional branch terminator in loop block");
            }

            let end_block = &func.blocks[2];
            assert_eq!(end_block.name, "end");
            assert_eq!(end_block.params.len(), 1);
            assert_eq!(end_block.params[0], ("i".to_string(), BaseType::I32));
            assert!(end_block.instructions.is_empty());

            if let Terminator::Return(Some(val)) = &end_block.terminator {
                assert_eq!(*val, Value::Register("i".to_string()));
            } else {
                panic!("Expected return terminator in end block");
            }
        } else {
            panic!("Expected function definition");
        }
    }

    #[test]
    fn test_select_instruction() {
        let input = r#"
fn @test_select(%cond: i32, %a: i32, %b: i32) -> i32 {
start:
    %result = select.i32 %cond, %a, %b
    ret %result
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];
            if let Instruction::Select {
                dest,
                ty,
                condition,
                true_val,
                false_val,
            } = &block.instructions[0]
            {
                assert_eq!(dest, "result");
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(*condition, Value::Register("cond".to_string()));
                assert_eq!(*true_val, Value::Register("a".to_string()));
                assert_eq!(*false_val, Value::Register("b".to_string()));
            } else {
                panic!("Expected select instruction");
            }
        }
    }

    #[test]
    fn test_store_instruction() {
        let input = r#"
fn @test_store() {
start:
    %ptr = alloc.i32 1
    %val = add.i32 10, 20
    store.i32 %ptr, %val
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];

            if let Instruction::Memory {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[0]
            {
                assert_eq!(dest, &Some("ptr".to_string()));
                assert_eq!(*op, MemOp::Alloc);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands, &vec![Value::Int(1)]);
            }

            // (address first, then value)
            if let Instruction::Memory {
                dest,
                op,
                ty,
                operands,
            } = &block.instructions[2]
            {
                assert_eq!(*dest, None);
                assert_eq!(*op, MemOp::Store);
                assert_eq!(*ty, BaseType::I32);
                assert_eq!(operands[0], Value::Register("ptr".to_string())); // address
                assert_eq!(operands[1], Value::Register("val".to_string())); // value
            }
        }
    }

    #[test]
    fn test_complex_block_parameters() {
        let input = r#"
fn @fibonacci(%n: i32) -> i32 {
start:
    %is_base = le.i32 %n, 1
    brif %is_base, base(%n), recurse(%n)

base(%val: i32):
    ret %val

recurse(%n: i32):
    %n_minus_1 = sub.i32 %n, 1
    %n_minus_2 = sub.i32 %n, 2
    %fib1 = call @fibonacci(%n_minus_1)
    %fib2 = call @fibonacci(%n_minus_2)
    %result = add.i32 %fib1, %fib2
    ret %result
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            assert_eq!(func.blocks.len(), 3);

            if let Terminator::BranchIf(condition, true_target, false_target) =
                &func.blocks[0].terminator
            {
                assert_eq!(*condition, Value::Register("is_base".to_string()));
                assert_eq!(true_target.block, "base");
                assert_eq!(true_target.args.len(), 1);
                assert_eq!(false_target.block, "recurse");
                assert_eq!(false_target.args.len(), 1);
            }
        }
    }

    #[test]
    fn test_shift_operations() {
        let input = r#"
fn @test_shifts(%val: i32, %amount: i32) {
start:
    %left_shift = lsl.i32 %val, %amount
    %right_shift = lsr.i32 %val, %amount
    %arith_shift = asr.i32 %val, %amount
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];
            let expected_ops = [ArithOp::Lsl, ArithOp::Lsr, ArithOp::Asr];

            for (i, expected_op) in expected_ops.iter().enumerate() {
                if let Instruction::Arith { op, operands, .. } = &block.instructions[i] {
                    assert_eq!(op, expected_op);
                    assert_eq!(operands.len(), 2);
                    assert_eq!(operands[0], Value::Register("val".to_string()));
                    assert_eq!(operands[1], Value::Register("amount".to_string()));
                }
            }
        }
    }

    #[test]
    fn test_mixed_global_register_literal_operands() {
        let input = r#"
data @constant: i32 = 42

fn @test_mixed(%param: i32) {
start:
    %result1 = add.i32 %param, @constant
    %result2 = add.i32 @constant, 100
    %result3 = mul.i32 %param, 5
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[1] {
            let block = &func.blocks[0];

            // register + global
            if let Instruction::Arith { operands, .. } = &block.instructions[0] {
                assert_eq!(operands[0], Value::Register("param".to_string()));
                assert_eq!(operands[1], Value::Global("constant".to_string()));
            }

            // global + literal
            if let Instruction::Arith { operands, .. } = &block.instructions[1] {
                assert_eq!(operands[0], Value::Global("constant".to_string()));
                assert_eq!(operands[1], Value::Int(100));
            }

            // register + literal
            if let Instruction::Arith { operands, .. } = &block.instructions[2] {
                assert_eq!(operands[0], Value::Register("param".to_string()));
                assert_eq!(operands[1], Value::Int(5));
            }
        }
    }

    #[test]
    fn test_signed_vs_unsigned_operations() {
        let input = r#"
fn @test_signed_unsigned(%a: i32, %b: i32) {
start:
    %div_signed = div.i32 %a, %b
    %div_unsigned = udiv.i32 %a, %b
    %rem_signed = rem.i32 %a, %b
    %rem_unsigned = urem.i32 %a, %b
    %cmp_signed = lt.i32 %a, %b
    %cmp_unsigned = ult.i32 %a, %b
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];

            let expected_arith_ops = [ArithOp::Div, ArithOp::Udiv, ArithOp::Rem, ArithOp::Urem];
            let expected_cmp_ops = [CmpOp::Lt, CmpOp::Ult];

            for (i, expected_op) in expected_arith_ops.iter().enumerate() {
                if let Instruction::Arith { op, .. } = &block.instructions[i] {
                    assert_eq!(op, expected_op);
                }
            }

            for (i, expected_op) in expected_cmp_ops.iter().enumerate() {
                if let Instruction::Compare { op, .. } = &block.instructions[4 + i] {
                    assert_eq!(op, expected_op);
                }
            }
        }
    }

    #[test]
    fn test_float_promotion_demotion() {
        let input = r#"
fn @test_float_conversions(%f32_val: f32, %f64_val: f64) {
start:
    %promoted = fpromote.f64 %f32_val
    %demoted = fdemote.f32 %f64_val
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];

            if let Instruction::Convert { op, dest_ty, .. } = &block.instructions[0] {
                assert_eq!(*op, ConvOp::Fpromote);
                assert_eq!(*dest_ty, BaseType::F64);
            }

            if let Instruction::Convert { op, dest_ty, .. } = &block.instructions[1] {
                assert_eq!(*op, ConvOp::Fdemote);
                assert_eq!(*dest_ty, BaseType::F32);
            }
        }
    }

    #[test]
    fn test_pointer_integer_conversions() {
        let input = r#"
fn @test_ptr_conversions(%ptr_val: ptr, %int_val: i64) {
start:
    %ptr_to_int = ptoi.i64 %ptr_val
    %int_to_ptr = itop.i64 %int_val
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[0] {
            let block = &func.blocks[0];

            if let Instruction::Convert { op, dest_ty, .. } = &block.instructions[0] {
                assert_eq!(*op, ConvOp::Ptoi);
                assert_eq!(*dest_ty, BaseType::I64);
            }

            if let Instruction::Convert { op, .. } = &block.instructions[1] {
                assert_eq!(*op, ConvOp::Itop);
            }
        }
    }

    #[test]
    fn test_empty_blocks_and_calls() {
        let input = r#"
declare fn @no_params() -> i32
declare fn @void_func()

fn @test() {
start:
    %val1 = call @no_params()
    call @void_func()
    br end
end:
    ret
}
"#;
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer).unwrap();
        let program = parser.parse().unwrap();

        if let Definition::Function(func) = &program.definitions[2] {
            let block = &func.blocks[0];

            if let Instruction::Call {
                dest,
                function,
                args,
            } = &block.instructions[0]
            {
                assert_eq!(dest, &Some("val1".to_string()));
                assert_eq!(function, "no_params");
                assert!(args.is_empty());
            }

            if let Terminator::Branch(target) = &block.terminator {
                assert_eq!(target.block, "end");
                assert!(target.args.is_empty());
            }

            let end_block = &func.blocks[1];
            assert_eq!(end_block.name, "end");
            assert!(end_block.params.is_empty());
        }
    }
}

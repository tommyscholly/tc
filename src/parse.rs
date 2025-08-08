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
            MemOp::Load => 1,
            MemOp::Store => 2,
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
    use crate::tests::test_utils::{TestFixture, error_utils, parse_utils};

    #[test]
    fn test_all_fixtures_parse() {
        let fixtures = TestFixture::all();
        for fixture in &fixtures {
            parse_utils::assert_parses(fixture);
        }
    }

    #[test]
    fn test_simple_data_parsing() {
        let fixture = TestFixture::load("simple_data");
        parse_utils::assert_definition_count(&fixture, 1);
        parse_utils::assert_has_data_definitions(&fixture, &["foo"]);

        let program = fixture.parse().unwrap();
        if let Definition::Data(data) = &program.definitions[0] {
            assert_eq!(data.name, "foo");
            assert_eq!(data.value, GlobalData::Int(42));
        }
    }

    #[test]
    fn test_hello_world_parsing() {
        let fixture = TestFixture::load("hello_world");
        parse_utils::assert_definition_count(&fixture, 3);
        parse_utils::assert_has_data_definitions(&fixture, &["hello_world_z"]);
        parse_utils::assert_has_extern_function_definitions(&fixture, &["puts"]);
        parse_utils::assert_has_function_definitions(&fixture, &["main"]);
        parse_utils::assert_function_block_count(&fixture, "main", 1);
    }

    #[test]
    fn test_multi_block_function() {
        let fixture = TestFixture::load("multi_block_fn");
        parse_utils::assert_definition_count(&fixture, 1);
        parse_utils::assert_has_function_definitions(&fixture, &["example"]);
        parse_utils::assert_function_block_count(&fixture, "example", 3);

        let program = fixture.parse().unwrap();
        let func = parse_utils::get_function(&program, "example").unwrap();

        assert!(matches!(func.blocks[0].terminator, Terminator::Branch(_)));
        assert!(matches!(
            func.blocks[1].terminator,
            Terminator::BranchIf(_, _, _)
        ));
        assert!(matches!(func.blocks[2].terminator, Terminator::Return(_)));
    }

    #[test]
    fn test_select_and_store_instructions() {
        let select_fixture = TestFixture::load("select_instruction");
        let store_fixture = TestFixture::load("store_instruction");

        // Both should parse successfully
        parse_utils::assert_parses(&select_fixture);
        parse_utils::assert_parses(&store_fixture);

        // Verify select instruction structure
        let select_program = select_fixture.parse().unwrap();
        if let Definition::Function(func) = &select_program.definitions[0] {
            if let Instruction::Select { dest, ty, .. } = &func.blocks[0].instructions[0] {
                assert_eq!(dest, "result");
                assert_eq!(*ty, BaseType::I32);
            }
        }
    }

    #[test]
    fn test_various_arithmetic_operations() {
        let shift_fixture = TestFixture::load("shift_ops");
        let signed_unsigned_fixture = TestFixture::load("signed_unsigned_ops");

        parse_utils::assert_parses(&shift_fixture);
        parse_utils::assert_parses(&signed_unsigned_fixture);

        // Verify shift operations
        let shift_program = shift_fixture.parse().unwrap();
        if let Definition::Function(func) = &shift_program.definitions[0] {
            let expected_ops = [ArithOp::Lsl, ArithOp::Lsr, ArithOp::Asr];
            for (i, expected_op) in expected_ops.iter().enumerate() {
                if let Instruction::Arith { op, .. } = &func.blocks[0].instructions[i] {
                    assert_eq!(op, expected_op);
                }
            }
        }
    }

    #[test]
    fn test_conversion_operations() {
        let float_fixture = TestFixture::load("float_conversions");
        let pointer_fixture = TestFixture::load("pointer_conversions");

        parse_utils::assert_parses(&float_fixture);
        parse_utils::assert_parses(&pointer_fixture);
    }

    #[test]
    fn test_error_cases() {
        // Test missing terminator
        let invalid_syntax = r#"
fn @test() -> i32 {
start:
    %a = add.i32 10, 20
    %b = add.i32 5, 7
}"#;
        error_utils::assert_parse_error(invalid_syntax, "Expected terminator");
    }
}

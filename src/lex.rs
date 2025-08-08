use anyhow::{Result, anyhow};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Data,
    Fn,
    Declare,
    Type,
    Br,
    Brif,
    Ret,
    Call,

    // Base types
    I8,
    I32,
    I64,
    F32,
    F64,
    Ptr,

    // Arithmetic operations
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

    // Memory operations
    Load,
    Store,
    Alloc,

    // Comparison operations
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

    // Other operations
    Select,
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

    // Identifiers and literals
    Ident(String),
    Global(String),   // @identifier
    Register(String), // %identifier
    // UserType(String), // ^identifier
    Int(i64),
    Float(f64),
    String(String),
    Number(u64),

    // Symbols
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    Arrow,
    Equals,

    // Special
    Comment(String),
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "IDENT({})", s),
            Token::Global(s) => write!(f, "@{}", s),
            Token::Register(s) => write!(f, "%{}", s),
            // Token::UserType(s) => write!(f, "^{}", s),
            Token::Int(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Number(n) => write!(f, "{}", n),
            Token::Comment(s) => write!(f, "#{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.first().copied();

        Lexer {
            input: chars,
            position: 0,
            current_char,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<String> {
        let mut result = String::new();
        self.advance(); // Skip opening quote

        while let Some(ch) = self.current_char {
            match ch {
                '"' => {
                    self.advance(); // Skip closing quote
                    return Ok(result);
                }
                '\\' => {
                    self.advance();
                    match self.current_char {
                        Some('n') => result.push('\n'),
                        Some('t') => result.push('\t'),
                        Some('r') => result.push('\r'),
                        Some('\\') => result.push('\\'),
                        Some('"') => result.push('"'),
                        Some('0') => {
                            if self.peek() == Some('0') {
                                self.advance();
                                result.push('\0');
                            } else {
                                result.push('\0');
                            }
                        }
                        Some(c) => result.push(c),
                        None => return Err(anyhow!("Unterminated string escape")),
                    }
                    self.advance();
                }
                c => {
                    result.push(c);
                    self.advance();
                }
            }
        }

        Err(anyhow!("Unterminated string"))
    }

    fn read_number(&mut self) -> Token {
        let mut result = String::new();
        let mut is_negative = false;

        if self.current_char == Some('-') {
            is_negative = true;
            result.push('-');
            self.advance();
        }

        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Check for float
        if self.current_char == Some('.') && self.peek().is_some_and(|c| c.is_ascii_digit()) {
            result.push('.');
            self.advance();

            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    result.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }

            // Handle scientific notation
            if self.current_char == Some('e') || self.current_char == Some('E') {
                result.push(self.current_char.unwrap());
                self.advance();

                if self.current_char == Some('+') || self.current_char == Some('-') {
                    result.push(self.current_char.unwrap());
                    self.advance();
                }

                while let Some(ch) = self.current_char {
                    if ch.is_ascii_digit() {
                        result.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }

            Token::Float(result.parse().unwrap_or(0.0))
        } else {
            Token::Int(result.parse().unwrap_or(0))
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();

        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        result
    }

    fn read_comment(&mut self) -> String {
        let mut result = String::new();
        self.advance(); // Skip '#'

        while let Some(ch) = self.current_char {
            if ch == '\n' {
                break;
            }
            result.push(ch);
            self.advance();
        }

        result
    }

    fn keyword_or_ident(&self, ident: &str) -> Token {
        match ident {
            "data" => Token::Data,
            "fn" => Token::Fn,
            "declare" => Token::Declare,
            "type" => Token::Type,
            "br" => Token::Br,
            "brif" => Token::Brif,
            "ret" => Token::Ret,
            "call" => Token::Call,
            "i8" => Token::I8,
            "i32" => Token::I32,
            "i64" => Token::I64,
            "f32" => Token::F32,
            "f64" => Token::F64,
            "ptr" => Token::Ptr,
            "add" => Token::Add,
            "sub" => Token::Sub,
            "mul" => Token::Mul,
            "div" => Token::Div,
            "udiv" => Token::Udiv,
            "rem" => Token::Rem,
            "urem" => Token::Urem,
            "and" => Token::And,
            "or" => Token::Or,
            "xor" => Token::Xor,
            "lsl" => Token::Lsl,
            "lsr" => Token::Lsr,
            "asr" => Token::Asr,
            "neg" => Token::Neg,
            "load" => Token::Load,
            "store" => Token::Store,
            "alloc" => Token::Alloc,
            "eq" => Token::Eq,
            "ne" => Token::Ne,
            "lt" => Token::Lt,
            "le" => Token::Le,
            "gt" => Token::Gt,
            "ge" => Token::Ge,
            "ult" => Token::Ult,
            "ule" => Token::Ule,
            "ugt" => Token::Ugt,
            "uge" => Token::Uge,
            "select" => Token::Select,
            "sext" => Token::Sext,
            "zext" => Token::Zext,
            "trunc" => Token::Trunc,
            "itof" => Token::Itof,
            "uitof" => Token::Uitof,
            "ftoi" => Token::Ftoi,
            "fpromote" => Token::Fpromote,
            "fdemote" => Token::Fdemote,
            "ptoi" => Token::Ptoi,
            "itop" => Token::Itop,
            "bitcast" => Token::Bitcast,
            _ => Token::Ident(ident.to_string()),
        }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        match self.current_char {
            None => Ok(Token::Eof),
            Some('(') => {
                self.advance();
                Ok(Token::LeftParen)
            }
            Some(')') => {
                self.advance();
                Ok(Token::RightParen)
            }
            Some('{') => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            Some('}') => {
                self.advance();
                Ok(Token::RightBrace)
            }
            Some('[') => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            Some(']') => {
                self.advance();
                Ok(Token::RightBracket)
            }
            Some(',') => {
                self.advance();
                Ok(Token::Comma)
            }
            Some(':') => {
                self.advance();
                Ok(Token::Colon)
            }
            Some(';') => {
                self.advance();
                Ok(Token::Semicolon)
            }
            Some('.') => {
                self.advance();
                Ok(Token::Dot)
            }
            Some('=') => {
                self.advance();
                Ok(Token::Equals)
            }
            Some('-') => {
                if self.peek() == Some('>') {
                    self.advance();
                    self.advance();
                    Ok(Token::Arrow)
                } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    return Ok(self.read_number());
                } else {
                    return Err(anyhow!("Unexpected character: -"));
                }
            }
            Some('@') => {
                self.advance();
                let ident = self.read_identifier();
                Ok(Token::Global(ident))
            }
            Some('%') => {
                self.advance();
                let ident = self.read_identifier();
                Ok(Token::Register(ident))
            }
            Some('^') => {
                panic!("User types are not yet supported");
                // self.advance();
                // let ident = self.read_identifier();
                // Ok(Token::UserType(ident))
            }
            Some('"') => {
                let string_val = self.read_string()?;
                Ok(Token::String(string_val))
            }
            Some('#') => {
                let comment = self.read_comment();
                Ok(Token::Comment(comment))
            }
            Some(ch) if ch.is_ascii_digit() => Ok(self.read_number()),
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                Ok(self.keyword_or_ident(&ident))
            }
            Some(ch) => Err(anyhow!("Unexpected character: {}", ch)),
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = matches!(token, Token::Eof);
            tokens.push(token);

            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_utils::{TestFixture, error_utils, lex_utils};
    use std::f64::consts::PI;

    #[test]
    fn test_all_fixtures_tokenize() {
        let fixtures = TestFixture::all();
        for fixture in &fixtures {
            lex_utils::assert_tokenizes(fixture);
        }
    }

    #[test]
    fn test_hello_world_tokens() {
        let fixture = TestFixture::load("hello_world");
        lex_utils::assert_contains_tokens(
            &fixture,
            &[
                Token::Data,
                Token::Global("hello_world_z".to_string()),
                Token::String("Hello, World\0".to_string()),
                Token::Declare,
                Token::Fn,
                Token::Global("puts".to_string()),
                Token::Global("main".to_string()),
                Token::Register("r".to_string()),
                Token::Call,
            ],
        );
    }

    #[test]
    fn test_arithmetic_tokens() {
        let fixture = TestFixture::load("fn_with_entry");
        lex_utils::assert_contains_tokens(
            &fixture,
            &[
                Token::Register("result".to_string()),
                Token::Equals,
                Token::Add,
                Token::Dot,
                Token::I32,
            ],
        );
    }

    #[test]
    fn test_shift_operation_tokens() {
        let fixture = TestFixture::load("shift_ops");
        lex_utils::assert_contains_tokens(&fixture, &[Token::Lsl, Token::Lsr, Token::Asr]);
    }

    #[test]
    fn test_comparison_tokens() {
        let fixture = TestFixture::load("signed_unsigned_ops");
        lex_utils::assert_contains_tokens(
            &fixture,
            &[
                Token::Lt,
                Token::Ult,
                Token::Div,
                Token::Udiv,
                Token::Rem,
                Token::Urem,
            ],
        );
    }

    #[test]
    fn test_conversion_tokens() {
        let fixture = TestFixture::load("float_conversions");
        lex_utils::assert_contains_tokens(
            &fixture,
            &[
                Token::Fpromote,
                Token::Fdemote,
                Token::Itof,
                Token::Uitof,
                Token::Ftoi,
            ],
        );
    }

    #[test]
    fn test_manual_number_parsing() {
        let input = "42 -17 3.14 -2.5e10";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[0], Token::Int(42)));
        assert!(matches!(tokens[1], Token::Int(-17)));
        assert!(matches!(tokens[2], Token::Float(f) if (f - PI).abs() < 0.005));
        assert!(matches!(tokens[3], Token::Float(f) if (f + 2.5e10).abs() < 1e6));
    }

    #[test]
    fn test_error_cases() {
        // Test unterminated string
        error_utils::assert_lex_error(r#"data @test: [i8; 5] = "hello"#, "Unterminated string");

        // Test unexpected character
        error_utils::assert_lex_error("fn @test() { $ }", "Unexpected character");
    }
}

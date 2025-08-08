use crate::lex::{Lexer, Token};
use crate::parse::{Parser, Program};
use anyhow::Result;
use std::fs;
use std::path::Path;

pub struct TestFixture {
    pub name: &'static str,
    pub source: String,
}

impl TestFixture {
    pub fn load(name: &'static str) -> Self {
        let fixture_path = format!("src/tests/fixtures/{}.tc", name);
        let source = fs::read_to_string(&fixture_path)
            .unwrap_or_else(|_| panic!("Failed to load test fixture: {}", fixture_path));

        TestFixture { name, source }
    }

    pub fn all() -> Vec<TestFixture> {
        vec![
            TestFixture::load("hello_world"),
            TestFixture::load("simple_data"),
            TestFixture::load("empty_fn"),
            TestFixture::load("fn_with_entry"),
            TestFixture::load("two_functions"),
            TestFixture::load("multi_block_fn"),
            TestFixture::load("select_instruction"),
            TestFixture::load("store_instruction"),
            TestFixture::load("complex_block_params"),
            TestFixture::load("shift_ops"),
            TestFixture::load("mixed_operands"),
            TestFixture::load("signed_unsigned_ops"),
            TestFixture::load("float_conversions"),
            TestFixture::load("pointer_conversions"),
            TestFixture::load("empty_blocks_calls"),
        ]
    }

    pub fn tokenize(&self) -> Result<Vec<Token>> {
        let mut lexer = Lexer::new(&self.source);
        lexer.tokenize()
    }

    pub fn parse(&self) -> Result<Program> {
        let lexer = Lexer::new(&self.source);
        let mut parser = Parser::new(lexer)?;
        parser.parse()
    }

    /// source without comments and extra whitespace
    pub fn normalized_source(&self) -> String {
        self.source
            .lines()
            .map(|line| {
                if let Some(comment_pos) = line.find('#') {
                    line[..comment_pos].trim()
                } else {
                    line.trim()
                }
            })
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub mod lex_utils {
    use super::*;

    pub fn assert_tokenizes(fixture: &TestFixture) {
        fixture
            .tokenize()
            .unwrap_or_else(|e| panic!("Failed to tokenize {}: {}", fixture.name, e));
    }

    pub fn assert_contains_tokens(fixture: &TestFixture, expected_tokens: &[Token]) {
        let tokens = fixture
            .tokenize()
            .unwrap_or_else(|e| panic!("Failed to tokenize {}: {}", fixture.name, e));

        for expected_token in expected_tokens {
            assert!(
                tokens.contains(expected_token),
                "Token {:?} not found in fixture '{}'. Actual tokens: {:?}",
                expected_token,
                fixture.name,
                tokens
            );
        }
    }

    pub fn count_token(fixture: &TestFixture, token: &Token) -> usize {
        let tokens = fixture
            .tokenize()
            .unwrap_or_else(|e| panic!("Failed to tokenize {}: {}", fixture.name, e));
        tokens.iter().filter(|&t| t == token).count()
    }
}

pub mod parse_utils {
    use super::*;
    use crate::parse::*;

    pub fn assert_parses(fixture: &TestFixture) {
        fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));
    }

    pub fn assert_definition_count(fixture: &TestFixture, expected_count: usize) {
        let program = fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));

        assert_eq!(
            program.definitions.len(),
            expected_count,
            "Expected {} definitions in '{}', found {}",
            expected_count,
            fixture.name,
            program.definitions.len()
        );
    }

    pub fn assert_has_data_definitions(fixture: &TestFixture, expected_names: &[&str]) {
        let program = fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));

        let data_names: Vec<&str> = program
            .definitions
            .iter()
            .filter_map(|def| match def {
                Definition::Data(data) => Some(data.name.as_str()),
                _ => None,
            })
            .collect();

        for expected_name in expected_names {
            assert!(
                data_names.contains(expected_name),
                "Data definition '{}' not found in fixture '{}'. Found: {:?}",
                expected_name,
                fixture.name,
                data_names
            );
        }
    }

    pub fn assert_has_function_definitions(fixture: &TestFixture, expected_names: &[&str]) {
        let program = fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));

        let function_names: Vec<&str> = program
            .definitions
            .iter()
            .filter_map(|def| match def {
                Definition::Function(func) => Some(func.name.as_str()),
                _ => None,
            })
            .collect();

        for expected_name in expected_names {
            assert!(
                function_names.contains(expected_name),
                "Function definition '{}' not found in fixture '{}'. Found: {:?}",
                expected_name,
                fixture.name,
                function_names
            );
        }
    }

    pub fn assert_has_extern_function_definitions(fixture: &TestFixture, expected_names: &[&str]) {
        let program = fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));

        let extern_names: Vec<&str> = program
            .definitions
            .iter()
            .filter_map(|def| match def {
                Definition::ExternFunction(func) => Some(func.name.as_str()),
                _ => None,
            })
            .collect();

        for expected_name in expected_names {
            assert!(
                extern_names.contains(expected_name),
                "Extern function definition '{}' not found in fixture '{}'. Found: {:?}",
                expected_name,
                fixture.name,
                extern_names
            );
        }
    }

    pub fn get_function<'a>(program: &'a Program, name: &str) -> Option<&'a Function> {
        program.definitions.iter().find_map(|def| match def {
            Definition::Function(func) if func.name == name => Some(func),
            _ => None,
        })
    }

    pub fn assert_function_block_count(
        fixture: &TestFixture,
        function_name: &str,
        expected_count: usize,
    ) {
        let program = fixture
            .parse()
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", fixture.name, e));

        let function = get_function(&program, function_name).unwrap_or_else(|| {
            panic!(
                "Function '{}' not found in fixture '{}'",
                function_name, fixture.name
            )
        });

        assert_eq!(
            function.blocks.len(),
            expected_count,
            "Expected {} blocks in function '{}' in fixture '{}', found {}",
            expected_count,
            function_name,
            fixture.name,
            function.blocks.len()
        );
    }
}

pub mod error_utils {
    use super::*;

    pub fn assert_lex_error(source: &str, expected_error_substring: &str) {
        let mut lexer = Lexer::new(source);
        match lexer.tokenize() {
            Ok(_) => panic!("Expected lexing to fail, but it succeeded"),
            Err(e) => assert!(
                e.to_string().contains(expected_error_substring),
                "Expected error to contain '{}', but got: {}",
                expected_error_substring,
                e
            ),
        }
    }

    pub fn assert_parse_error(source: &str, expected_error_substring: &str) {
        let lexer = Lexer::new(source);
        let mut parser =
            Parser::new(lexer).expect("Parser creation should succeed for parse error tests");

        match parser.parse() {
            Ok(_) => panic!("Expected parsing to fail, but it succeeded"),
            Err(e) => assert!(
                e.to_string().contains(expected_error_substring),
                "Expected error to contain '{}', but got: {}",
                expected_error_substring,
                e
            ),
        }
    }
}

// meta tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_loading() {
        let fixture = TestFixture::load("hello_world");
        assert_eq!(fixture.name, "hello_world");
        assert!(!fixture.source.is_empty());
        assert!(fixture.source.contains("@hello_world_z"));
    }

    #[test]
    fn test_all_fixtures_load() {
        let fixtures = TestFixture::all();
        assert!(!fixtures.is_empty());

        for fixture in &fixtures {
            lex_utils::assert_tokenizes(fixture);
        }
    }

    #[test]
    fn test_all_fixtures_parse() {
        let fixtures = TestFixture::all();

        for fixture in &fixtures {
            parse_utils::assert_parses(fixture);
        }
    }

    #[test]
    fn test_hello_world_contains_expected_tokens() {
        let fixture = TestFixture::load("hello_world");
        lex_utils::assert_contains_tokens(
            &fixture,
            &[
                Token::Data,
                Token::Global("hello_world_z".to_string()),
                Token::Declare,
                Token::Fn,
                Token::Global("puts".to_string()),
                Token::Global("main".to_string()),
            ],
        );
    }

    #[test]
    fn test_hello_world_has_expected_definitions() {
        let fixture = TestFixture::load("hello_world");
        parse_utils::assert_definition_count(&fixture, 3); // data + extern fn + fn
        parse_utils::assert_has_data_definitions(&fixture, &["hello_world_z"]);
        parse_utils::assert_has_extern_function_definitions(&fixture, &["puts"]);
        parse_utils::assert_has_function_definitions(&fixture, &["main"]);
    }
}

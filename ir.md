# Intermediate Representation

This document describes the intermediate representation of this compiler. It is a SSA form, three-address code intermediate representation.

## Input Format

Currently, only a text-based format is supported. An input file is defined by a sequence of #Definitions. TC will parse the file and output a resulting .s for the target architecture.

### Sigils

Just like LLVM, Cranelift, and QBE, TC uses sigils to avoid name conflicts and to quickly spot the nature of a given identifier.

- `@` prefixes a global variable, such as a function or data.
- `%` prefixes a function-scope register, such as a variable or parameter.
- `^` prefixes a user-defined type.
- blocks are suffixed by `:`

### Hello World

```llvmir
data @hello_world_z: [i8; 14] = "Hello, World\00"

declare fn @puts(ptr) -> i32

fn @main() -> i32 {
start:
    %r = call @puts(@hello_world_z) # call puts with the address of hello_world
    ret %r
}
```

## Types

```
BASE_TYPE ::= i8 | i32 | i64 | f32 | f64 | ptr
```

A `ptr` is a pointer type that is the size of a word on the target architecture. For instance, on ARM64, a `ptr` is 8 bytes.
It is preferred to use `ptr` instead of `i64` for pointers for readability. `i8` exists to represent a byte, primarily for string literals.
Eventually the plan is to support user-defined types, but this is not currently supported. That said, the planned syntax is:

```
# UNSUPPORTED
TYPE_DEF ::= 'type' '^' IDENT '=' '{' ( SUBTY ), '}'
```

With this, we could then define a type `Point` with two fields `x` and `y` of type `i32`, we would write:

```
type ^Point = { i32, i32 }
```

## Definitions

Data definitions are defined with the `data` keyword. They are declared with a name and must have an annotated type.
The annotate type can support an array for defining a string of bytes, but this is not supported for variables.
When these are used as values, they implicitly take the address of the data.

```
GLOBAL ::= '@' IDENT
DATA_DEF ::= 'data' GLOBAL ':' (BASE_TYPE | '[' BASE_TYPE ';' NUMBER ']') '=' STRING
```

Function definitions are defined with the `fn` keyword. Return types are specified with `->` and parameters are specified with `(` and `)`.

```
BLOCK_FN_PARAM ::= %IDENT ':' BASE_TYPE

FN_DEF ::=
    'fn' GLOBAL '(' (BLOCK_FN_PARAM), ')' (-> BASE_TYPE)?
    '{'
        BLOCK+
    '}'
```

External function definitions are defined with the `declare` keyword. They follow the same syntax as function definitions, but do not have a body.

```llvmir
EXTERN_FN_DEF ::= 'declare' 'fn' GLOBAL '(' (BASE_TYPE), ')' (-> BASE_TYPE)?
```

Basic blocks define the structure of a function. A function must have at least one basic block, and that entry block must take in the parameters of the function.
Blocks have a terminator at the end, that either branches to another block within the function, or terminates by returning (optionally with a value). 
The entry block is a special case, and should not be branched to. Either call the function recursively, or define a basic block that is under the entry block to branch to.
Instructions are terminated with a new line, but we omit these newlines in the BNF form for brevity.

```
BLOCK ::=
    IDENT ('(' (BLOCK_FN_PARAM), ')')?
    INST*
    TERM

```

Terminators support unconditional branches, conditional branches, and returns. 
`brif` expects that the value within the `REGISTER` is of size `i32`.

```
REGISTER ::= %IDENT
BRANCH_PARAM ::= REGISTER
BRANCH_TARGET ::= IDENT ( '(' BRANCH_PARAM (',' BRANCH_PARAM)* ')' )?

TERM ::=
    'br' BRANCH_TARGET
    'brif' VALUE ',' BRANCH_TARGET ',' BRANCH_TARGET # if VALUE, branch to BRANCH_TARGET, else BRANCH_TARGET
    'ret' (VALUE)?

```

## Instructions

Instructions are the building blocks of the IR. This IR uses a [three-address code](https://en.wikipedia.org/wiki/Three-address_code) where applicable.
Most instructions will have a destination register, and two source registers. Instructions can be polymorphic on the type of the operands.
For instance, the `add` instruction can take two `i32` operands, or two `f32` operands. These instructions are annotated with the type of the operands, e.g. `add.i32`.
Note that even though the BNF form implies that the destination register is not required, it is required for all instructions that yield values.

```
INST ::= (REGISTER '=')? (ARITH | MEM | CMP | CALL | CONV | SELECT)
```

Values are the building blocks of all instructions. A value is either a register, a constant, or a global value.

```
CONST ::= INT | FLOAT
VALUE ::= REGISTER | CONST | GLOBAL
```

Arithmetic instructions are the most common instructions in the IR. 
We abandon the BNF form in favor of listing each instruction explicitly, alongside the types that are supported.
Both registers are of the same type. Immediate values may be supported in the future, but are not currently.
The return value is always the same type as the instruction.

- `add.(i32|i64|f32|f64) VALUE, VALUE`
- `sub.(i32|i64|f32|f64) VALUE, VALUE`
- `mul.(i32|i64|f32|f64) VALUE, VALUE`
- `div.(i32|i64|f32|f64) VALUE, VALUE`
    - The div instruction assumes signed division.
- `neg.(i32|i64) VALUE`
- `rem.(i32|i64) VALUE, VALUE`
- `udiv.(i32|i64) VALUE, VALUE`
- `urem.(i32|i64) VALUE, VALUE`
- `and.(i32|i64) VALUE, VALUE`
- `or.(i32|i64) VALUE, VALUE`
- `xor.(i32|i64) VALUE, VALUE`
- `lsl.(i32|i64) VALUE, VALUE`
    - The shift instructions shift left or right by the amount of the second operand.
- `lsr.(i32|i64) VALUE, VALUE`
- `asr.(i32|i64) VALUE, VALUE`
    - Preserves the sign bit of the first operand.

Memory instructions are used to load and store values from memory.
Stores are annotated with the type of the value being stored, loads are annontated with the type of the value being loaded.
Both expect a register that contains a pointer to an addressable memory location. We may support a ptr with an offset in the future.

- `load.(i8|i32|i64|f32|f64) VALUE`
    - Returns the value at the address pointed to by the pointer.
- `store.(i8|i32|i64|f32|f64) VALUE, VALUE`
    - Stores the value at the address pointed to by the pointer.
    - The pointer is the first operand, and the value is the second operand.

We also support stack allocation via the `alloc` instruction. 
These are annotated with a type that is used to determine the size of the allocation, and the number of elements.

- `alloc.(i8|i32|i64|f32|f64) NUMBER`
    - This always returns a pointer to the start of the stack allocation.
    - We guarantee that this will be suitably aligned for the type.

Comparison instructions are used to compare two values. They return always return a `i32` value, and are annotated with the type of the value being compared.

- `eq.(i32|i64|f32|f64) VALUE, VALUE`
- `ne.(i32|i64|f32|f64) VALUE, VALUE`
- `lt.(i32|i64|f32|f64) VALUE, VALUE`
- `le.(i32|i64|f32|f64) VALUE, VALUE`
- `gt.(i32|i64|f32|f64) VALUE, VALUE`
- `ge.(i32|i64|f32|f64) VALUE, VALUE`

There are also unsigned comparison variants.

- `ult.(i32|i64) VALUE, VALUE`
- `ule.(i32|i64) VALUE, VALUE`
- `ugt.(i32|i64) VALUE, VALUE`
- `uge.(i32|i64) VALUE, VALUE`

There is also a `select` instruction that allows conditional selection without branching. 
The annotated type is the type of the value being selected, the second and third operands.
Just like with `brif`, the condition is always a `i32` value.

- `select.(i32|i64|f32|f64) VALUE, VALUE, VALUE`

Conversion instructions are used to change the type or size of values. 
The source type is inferred from the operand, and the destination type is explicitly annotated. 
These instructions handle both size changes within the same type family (integer widening/narrowing) and conversions between different type families (integer to float, etc.).

### Integer Size Conversions

- `sext.(i32|i64) VALUE`
- `zext.(i32|i64) VALUE`
- `trunc.(i8|i32) VALUE`

### Integer-Float Conversions

- `itof.(f32|f64) VALUE`
- `uitof.(f32|f64) VALUE`
- `ftoi.(i32|i64) VALUE`

### Float Size Conversions

- `fpromote.f64 VALUE`
- `fdemote.f32 VALUE`

### Pointer Conversions

We may eventually support direct pointer arithmetic, rather than requiring to convert to an integer first.

- `ptoi.(i32|i64) VALUE`
- `itop VALUE`

### Bitcast

- `bitcast.(i32|i64|f32|f64|ptr) VALUE`
    - Reinterpret the bits of a value as a different type without changing the bit pattern.
    - Both source and destination types must be the same size.
    - Example: `%bits = bitcast.i32 %float_f32` reinterprets float bits as integer.

Calls are used to call procedures. They break from the three-address code model, but are not annotated with a return type or parameter type(s).
If a function returns a value, it is a type error not to assign the value to a register. 
We return to BNF form to describe the call instruction.

```llvmir
CALL ::= 'call' GLOBAL '(' (VALUE (',' VALUE)*)? ')'
```

We currently do not support tail call instructions, or variadic function calls.

## SSA Form

The compiler assumes that the input is in SSA form.
Instead of Phi instructions, we use basic block parameters instead. This gives us the same ability as Phi instructions, but an (arguably) more intuitive syntax.
Eventually, we may investigate converting non-SSA IR to SSA form.

```llvmir
fn @example(%a: i32, %b: i32) -> i32 {
start:
    %x = add.i32 %a, %b
    br loop(%x)

loop(%i: i32):
    %cond = lt.i32 %i, 10
    %next = sub.i32 %i, 1
    brif %cond, loop(%next), end

end:
    ret %i
}
```

## Attributions

This intermediate representation is inspired by a combination of the [QBE](https://c9x.me/compile/) and [Cranelift](https://cranelift.dev) intermediate representations. 
Additionally, referenced how [LLVM IR](https://llvm.org/docs/LangRef.html) implements their `alloca` instruction.

## BNF Grammar

The BNF grammar includes all the definitions within this document, including ones that are not supported yet.

```bnf
PROGRAM ::= DEFINITION*

DEFINITION ::= DATA_DEF | FN_DEF | EXTERN_FN_DEF | TYPE_DEF

BASE_TYPE ::= 'i8' | 'i32' | 'i64' | 'f32' | 'f64' | 'ptr'

TYPE ::= BASE_TYPE | ARRAY_TYPE | USER_TYPE
ARRAY_TYPE ::= '[' BASE_TYPE ';' NUMBER ']'
USER_TYPE ::= '^' IDENT

TYPE_DEF ::= 'type' '^' IDENT '=' '{' (BASE_TYPE (',' BASE_TYPE)*)? '}'

GLOBAL ::= '@' IDENT
REGISTER ::= '%' IDENT

DATA_DEF ::= 'data' GLOBAL ':' (BASE_TYPE | ARRAY_TYPE) '=' STRING

BLOCK_FN_PARAM ::= REGISTER ':' BASE_TYPE

FN_DEF ::= 'fn' GLOBAL '(' (BLOCK_FN_PARAM (',' BLOCK_FN_PARAM)*)? ')' ('->' BASE_TYPE)? '{' BLOCK+ '}'

EXTERN_FN_DEF ::= 'declare' 'fn' GLOBAL '(' (BASE_TYPE (',' BASE_TYPE)*)? ')' ('->' BASE_TYPE)?

BLOCK ::= IDENT ('(' (BLOCK_FN_PARAM (',' BLOCK_FN_PARAM)*)? ')')? ':' INST* TERM

INST ::= (REGISTER '=')? (ARITH | MEM | CMP | CALL | CONV | SELECT)

VALUE ::= REGISTER | CONST | GLOBAL
CONST ::= INT | FLOAT

ARITH ::= ARITH_OP '.' BASE_TYPE VALUE (',' VALUE)?
ARITH_OP ::= 'add' | 'sub' | 'mul' | 'div' | 'udiv' | 'rem' | 'urem' 
           | 'and' | 'or' | 'xor' | 'lsl' | 'lsr' | 'asr' | 'neg'

MEM ::= LOAD | STORE | ALLOC
LOAD ::= 'load' '.' BASE_TYPE VALUE
STORE ::= 'store' '.' BASE_TYPE VALUE ',' VALUE
ALLOC ::= 'alloc' '.' BASE_TYPE NUMBER

CMP ::= CMP_OP '.' BASE_TYPE VALUE ',' VALUE
CMP_OP ::= 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge' | 'ult' | 'ule' | 'ugt' | 'uge'

SELECT ::= 'select' '.' BASE_TYPE VALUE ',' VALUE ',' VALUE

CONV ::= CONV_OP '.' BASE_TYPE VALUE
CONV_OP ::= 'sext' | 'zext' | 'trunc' | 'itof' | 'uitof' | 'ftoi' 
          | 'fpromote' | 'fdemote' | 'ptoi' | 'itop' | 'bitcast'

CALL ::= 'call' GLOBAL '(' (VALUE (',' VALUE)*)? ')'

TERM ::= 'br' BRANCH_TARGET
             | 'brif' VALUE BRANCH_TARGET BRANCH_TARGET
             | 'ret' VALUE?

BRANCH_TARGET ::= IDENT ('(' (VALUE (',' VALUE)*)? ')')?

IDENT ::= [a-zA-Z_][a-zA-Z0-9_]*
NUMBER ::= [0-9]+
INT ::= '-'? [0-9]+
FLOAT ::= '-'? [0-9]+ '.' [0-9]+ ([eE] [+-]? [0-9]+)?
STRING ::= '"' ([^"\\] | '\\' .)* '"'
```

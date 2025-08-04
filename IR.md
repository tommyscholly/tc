# Intermediate Representation

This document describes the intermediate representation of this compiler. It is inspired by a combination of the QBE IR and the Cranelift IR.

## Input Format

Currently, only a text-based format is supported. An input file is defined by a sequence of #Definitions. TC will parse the file and output a resulting .s for the target architecture.

### Sigils

Just like LLVM, Cranelift, and QBE, TC uses sigils to avoid name conflicts and to quickly spot the nature of a given identifier.

- `@` prefixes a global variable, such as a function or data.
- `%` prefices a function-scope register, such as a variable or parameter.
- blocks are suffixed by `:`

### Hello World

```llvmir
data @hello_world = bytes "Hello, World"

declare fn @puts(ptr) -> i32

fn @main() -> i32 {
    %r = call @puts(ptr @hello_world)
    ret 0
}
```

## Definitions



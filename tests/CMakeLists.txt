find_package(Threads REQUIRED)

llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm    LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4         LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt             LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org-dev/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-kv-cache.cpp LABEL behavior)
    target_include_directories(test-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp  LABEL behavior)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2      LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox   LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

endif()

llama_build(test-llama-archs.cpp)
if (GGML_CPU AND (NOT WIN32 OR NOT BUILD_SHARED_LIBS))
    llama_build(test-fp8.cpp)
    set(FP8_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models-fp8")
    file(MAKE_DIRECTORY "${FP8_MODEL_DIR}")
    llama_test(test-llama-archs NAME test-generate-fp8-models LABEL "main behavior backend"
        ARGS -a llama -s 1 -o "${FP8_MODEL_DIR}")
    set_tests_properties(test-generate-fp8-models PROPERTIES FIXTURES_SETUP fp8-models)
    llama_test(test-fp8 NAME test-fp8-requantize LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}")
    llama_test(test-fp8 NAME test-fp8-kv LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}" --kv)
    set_tests_properties(test-fp8-requantize test-fp8-kv PROPERTIES FIXTURES_REQUIRED fp8-models)
endif()

set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
file(MAKE_DIRECTORY "${MODEL_DIR}")

llama_test(
    test-llama-archs
    NAME test-generate-models
    LABEL "main behavior backend"
    ARGS -o "${MODEL_DIR}"
)
set_tests_properties(test-generate-models PROPERTIES
    FIXTURES_SETUP generate-models
)

# Target model for test-speculative-rejection-sampling
set(SPEC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-speculative-models/target")
file(MAKE_DIRECTORY "${SPEC_TARGET_DIR}")
llama_test(
    test-llama-archs
    NAME test-generate-speculative-target
    LABEL main
    ARGS -a llama -s 999 -o "${SPEC_TARGET_DIR}"
)
set_tests_properties(
    test-generate-speculative-target
    PROPERTIES FIXTURES_SETUP speculative-model
)

# To avoid recreating the model use the fixture exclude any option:
# -FA "^speculative-model$"
llama_build_and_test(test-speculative-rejection-sampling.cpp ARGS
    -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-speculative-rejection-sampling PROPERTIES FIXTURES_REQUIRED speculative-model)

# Test context state handling across all architectures, using the generated dummy models
llama_test(
    test-llama-context
    LABEL "main behavior backend"
    ARGS --models "${MODEL_DIR}"
)
# abort on graph reallocations that the initial reserve should have covered
set_tests_properties(test-llama-context PROPERTIES
    FIXTURES_REQUIRED generate-models
    ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
)

llama_build(test-fusion.cpp)

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
llama_test_cmd(python3
    NAME test-conversion-fp8
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_conversion_fp8.py -v
)
llama_test_cmd(python3
    NAME test-kv-cache-scales
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_kv_cache_scales.py -v
)
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DREPO=ggml-org-dev/stories15M
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)
llama_build_and_test(test-warmup-profile.cpp LABEL behavior)
target_include_directories(test-warmup-profile PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml Threads::Threads)

if (GGML_QAIRT AND GGML_QAIRT_TESTS)
    llama_build(test-qairt-smoke.cpp LINK ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt-smoke PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${QAIRT_SDK_ROOT}/include/QNN)
    llama_test(test-qairt-smoke NAME qairt-host ARGS --host-tests)
    llama_build(test-qairt.cpp LINK ggml ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt)
    llama_test(test-qairt NAME qairt-m1-host ARGS --host)
    llama_build(test-qairt-fp16.cpp LINK ggml ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt-fp16 PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${PROJECT_SOURCE_DIR}/ggml/src ${QAIRT_SDK_ROOT}/include/QNN)
    llama_test(test-qairt-fp16 NAME qairt-m2-fp16-host ARGS --host)
    llama_build(test-qairt-quantized-weights.cpp LINK llama ggml ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt-quantized-weights PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${QAIRT_SDK_ROOT}/include/QNN)
    llama_test(test-qairt-quantized-weights NAME qairt-quantized-weights-host ARGS --host)
    llama_test(test-qairt NAME qairt-m2-admission ARGS --m2-host)
    llama_build(test-qairt-ops.cpp LINK ggml ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${QAIRT_SDK_ROOT}/include/QNN)
    llama_test(test-qairt-ops NAME qairt-m2-ops-host ARGS --host)
    if (GGML_QAIRT_HW_TESTS)
        foreach(op matmul fc conv)
            llama_test(test-qairt-quantized-weights NAME qairt-quantized-weights-${op}-htp LABEL hardware ARGS --op ${op} --encoding bw4 --scales positive)
            set_tests_properties(qairt-quantized-weights-${op}-htp PROPERTIES RUN_SERIAL TRUE)
        endforeach()
        llama_test(test-qairt-ops NAME qairt-m2-ops-htp LABEL hardware)
        set_tests_properties(qairt-m2-ops-htp PROPERTIES RUN_SERIAL TRUE)
    endif()
    llama_build(test-qairt-ffn.cpp LINK llama ggml ggml-qairt-core ggml-base Threads::Threads)
    target_include_directories(test-qairt-ffn PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${QAIRT_SDK_ROOT}/include/QNN)
    if (GGML_QAIRT_HW_TESTS)
        llama_test(test-qairt-fp16 NAME qairt-q41-htp LABEL hardware ARGS --q41)
        set_tests_properties(qairt-q41-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_W4FP16=1;GGML_QAIRT_APP_WRITE_WEIGHTS=0;GGML_QAIRT_SHARE_WEIGHT_GRAPHS=0")
        set(GGML_QAIRT_TEST_MODEL "" CACHE FILEPATH "Optional local Qwen3 GGUF for QAIRT Q4_1 FFN tests")
        if (GGML_QAIRT_TEST_MODEL)
            foreach(tokens 1 23)
                llama_test(test-qairt-ffn NAME qairt-q41-ffn-${tokens}-htp LABEL "hardware model" ARGS --model ${GGML_QAIRT_TEST_MODEL} --m ${tokens})
                set_tests_properties(qairt-q41-ffn-${tokens}-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_W4FP16=1;GGML_QAIRT_APP_WRITE_WEIGHTS=0;GGML_QAIRT_SHARE_WEIGHT_GRAPHS=0")
            endforeach()
        endif()
    endif()
    llama_build(test-qairt-model.cpp LINK llama ggml ggml-qairt-core Threads::Threads)
    llama_test(test-qairt-model NAME qairt-m2-fp16-reference-host ARGS --fp16-reference-self-test)
    target_include_directories(test-qairt-model PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-qairt ${QAIRT_SDK_ROOT}/include/QNN)
    llama_test(test-qairt NAME qairt-m1-gpu-disabled ARGS --unavailable)
    llama_test(test-qairt NAME qairt-m1-cpu-disabled ARGS --unavailable)
    set_tests_properties(qairt-m1-gpu-disabled PROPERTIES
        ENVIRONMENT "GGML_QAIRT_DEVICE=GPU;GGML_QAIRT_LIBRARY_DIR=C:/QNN-unused")
    set_tests_properties(qairt-m1-cpu-disabled PROPERTIES
        ENVIRONMENT "GGML_QAIRT_DEVICE=CPU;GGML_QAIRT_LIBRARY_DIR=C:/QNN-unused")
    if (GGML_QAIRT_HW_TESTS)
        llama_test(test-qairt NAME qairt-native-runtime-htp LABEL hardware ARGS --runtime-path)
        set_tests_properties(qairt-native-runtime-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_VALIDATE=0")
        llama_test(test-qairt NAME qairt-m1-htp LABEL hardware)
        llama_test(test-qairt-fp16 NAME qairt-m2-fp16-htp LABEL hardware)
        set_tests_properties(qairt-m1-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_VALIDATE=1")
        set_tests_properties(qairt-m2-fp16-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_VALIDATE=1;GGML_QAIRT_W4FP16=0;GGML_QAIRT_APP_WRITE_WEIGHTS=0;GGML_QAIRT_SHARE_WEIGHT_GRAPHS=0")
        foreach(share RANGE 0 1)
            llama_test(test-qairt-fp16 NAME qairt-appwrite-share${share}-htp LABEL hardware ARGS --weight-reuse)
            set_tests_properties(qairt-appwrite-share${share}-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_W4FP16=0;GGML_QAIRT_APP_WRITE_WEIGHTS=1;GGML_QAIRT_SHARE_WEIGHT_GRAPHS=${share}")
        endforeach()
        llama_test(test-qairt-quantized-weights NAME qairt-appwrite-fp16-direct-htp LABEL hardware ARGS --fp16-only --dynamic-fp16 --tensor-v1)
        set_tests_properties(qairt-appwrite-fp16-direct-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_W4FP16=0;GGML_QAIRT_APP_WRITE_WEIGHTS=0;GGML_QAIRT_SHARE_WEIGHT_GRAPHS=0")
        llama_test(test-qairt-fp16 NAME qairt-w4fp16-htp LABEL hardware ARGS --w4fp16)
        set_tests_properties(qairt-w4fp16-htp PROPERTIES RUN_SERIAL TRUE ENVIRONMENT "GGML_QAIRT_W4FP16=1")
    endif()
    llama_test(test-qairt-smoke NAME qairt-capture-success ARGS --capture-success)
    llama_test(test-qairt-smoke NAME qairt-capture-failure ARGS --capture-failure)
    if (GGML_QAIRT_HW_TESTS)
        llama_test(test-qairt-smoke NAME qairt-htp LABEL hardware ARGS --hardware-test)
    endif()
endif()

llama_build_and_test(test-backend-tensor.cpp LINK ggml)

if (BUILD_SHARED_LIBS)
    llama_build_and_test(test-llama-mmap.cpp ${PROJECT_SOURCE_DIR}/src/llama-impl.cpp ${PROJECT_SOURCE_DIR}/src/llama-mmap.cpp LABEL behavior)
    target_compile_definitions(test-llama-mmap PRIVATE LLAMA_BUILD)
else()
    llama_build_and_test(test-llama-mmap.cpp LABEL behavior)
endif()

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

if (APPLE)
    llama_build(test-rset-release.cpp)
    llama_build_and_test(test-host-buffer-release.cpp LINK ggml)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

#pragma once

#include <string>

namespace EscapeSequence {

#ifdef PC3_NO_ANSI_COLORS

static inline std::string GRAY = "";
static inline std::string RED = "";
static inline std::string GREEN = "";
static inline std::string YELLOW = "";
static inline std::string BLUE = "";
static inline std::string ORANGE = "";

static inline std::string RESET = "";
static inline std::string BOLD = "";
static inline std::string UNDERLINE = "";

static inline std::string CLEAR_LINE = "\033[2K";
static inline std::string LINE_UP = "\033[A";

static inline std::string HIDE_CURSOR = "";
static inline std::string SHOW_CURSOR = "";

#else

static inline std::string GRAY = "\033[90m";
static inline std::string RED = "\033[91m";
static inline std::string GREEN = "\033[92m";
static inline std::string YELLOW = "\033[93m";
static inline std::string BLUE = "\033[94m";
static inline std::string ORANGE = "\033[38;2;128;59;3m";

static inline std::string RESET = "\033[0m";
static inline std::string BOLD = "\033[1m";
static inline std::string UNDERLINE = "\033[4m";

static inline std::string CLEAR_LINE = "\033[2K";
static inline std::string LINE_UP = "\033[A";

static inline std::string HIDE_CURSOR = "\033[?25l";
static inline std::string SHOW_CURSOR = "\033[?25h";

#endif
} // namespace EscapeSequence
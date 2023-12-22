#pragma once

namespace EscapeSequence {
    static inline auto GREY = "\033[90m";
    static inline auto RED = "\033[91m";
    static inline auto GREEN = "\033[92m";
    static inline auto YELLOW = "\033[93m";
    static inline auto BLUE = "\033[94m";

    static inline auto RESET = "\033[0m";
    static inline auto BOLD = "\033[1m";
    static inline auto UNDERLINE = "\033[4m";

    static inline auto CLEAR_LINE = "\033[2K";
    static inline auto LINE_UP = "\033[A";

    static inline auto HIDE_CURSOR = "\033[?25l";
    static inline auto SHOW_CURSOR = "\033[?25h";
}
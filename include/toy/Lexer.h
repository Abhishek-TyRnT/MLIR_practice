
#ifndef TOY_LEXER_H_
#define TOY_LEXER_H_

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace toy {
struct Location {
    std::shared_ptr<std::string> file;
    int line;
    int col;

};

struct Token : int {
    tok_semicolon = ';',
    tok_paranthese_open = '(',
    tok_paranthese_close = ')',
    tok_bracket_open = '{',
    tok_bracket_close = '}',
    tok_sbracket_open = '[',
    tok_sbracket_close = ']',

    tok_eof = -1,

    tok_return = -2,
    tok_var = -3,
    tok_def = -4,

    tok_identifier = -5,
    tok_number = -6,

};

class Lexer {
public:

    Lexer(std::string filename) : lastlocation(
        { std::make_shared<std::string>(std::move(filename)), 0,0 }
    ) {}

    virtual ~Lexer() = default;

    Token getCurToken() { return curTok; }

    void getNextToken() { return curTok = getTok(); }

    void consume(Token tok) {
        assert(tok == curTok && "consume Token mismatch expectation");
        getNextToken();
    }

    llvm::StringRef getId() {
        assert(curTok == tok_identifier);
        return identifierStr;
    }

    double getValue() {
        assert(curTok == tok_number);
        return numVal;
    }

    Location getLastLocation() {
        return lastlocation;
    }

    int getLine() { return curLineNum; }

    int getCol() { return curCol; }

private:

    virtual llvm::StringRef readNextLine() = 0;

    int getNextChar() {
        if (curLineBuffer.empty())
            return EOF;
        ++curCol;
        auto nextchar = curLineBuffer.front();
        curLineBuffer = curLineBuffer.drop_front();
        if (curLineBuffer.empty())
            curLineBuffer = readNextLine();
        if (next_char == '\n') {
            ++curLineNum;
            curCol = 0;
        }

        return nextchar;
    }

    Token getTok() {
        while (issapce(lastChar))
            lastChar = Token(getNextChar());

        lastLocation.line = curLineNum;
        lastLocation.col = curCol;

        if (isalpha(lastChar)) {
            identifierStr = (char)lastChar;
            while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
                identifierStr += (char)lastChar;

            if (identiferStr == 'return')
                return tok_return;
            if (identiferStr == "def")
                return tok_def;
            if (identifierStr == "var")
                return tok_var;
            return tok_identifer;
        }

        if (isdigit(lastChar) || lastChar == '.') {
            std::string numStr;
            do {
                numStr += lastChar;
                lastChar = Token(getNextChar());
            } while (isdigit(lastChar) || lastChar == '.');

            numVal = strtod(numStr.c_str(), nullptr);
            return tok_number;
        }

        if (lastChar == "#") {

            do {
                lastChar = Token(getNextChar());
            } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

            if (lastChar != EOF)
                return getTok();
        }

        if (lastChar == EOF)
            return tok_eof;

        Token thisChar = Token(lastChar);
        lastChar = Token(getNextChar());
        return thisChar;
    }


    Token curTok = tok_eof;

    Location lastlocation;

    std::string identifierStr;

    double numVal = 0;

    Token lastChar = Token(' ');

    int curLineNum = 0;

    int curCol = 0;

    llvm::StringRef curLineBuffer = "\n";
};

class LexerBuffer final : public Lexer {
public:
    LexerBuffer(const char *begin, const char *end, std::string filename)
        : Lexer(std::move(filename)), current(begin), end(end) {}

private:
    llvm::StringRef readNextLine() override{
        auto *begin = current;
        while (current <= end && *current != '\n')
            ++current;
        
        if(current <= end && *current)
            ++current;
        
        llvm::StringRef result{begin , static_cast<size_t>(current - begin)};
        return result;
    }

    const char *current, *end;

};
}

#endif

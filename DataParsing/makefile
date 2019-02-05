SOURCES = $(wildcard *.cpp) \
			$(wildcard Models/*.cpp) \
			$(wildcard Parsers/*.cpp)
HEADERS = $(wildcard *.h) $(wildcard Models/*.h) $(wildcard Parsers/*.cpp)

parser: $(SOURCES) $(HEADERS)
	g++ $(SOURCES) -o dataparser -std=c++11
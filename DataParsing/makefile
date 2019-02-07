SOURCES = $(wildcard *.cpp) \
			$(wildcard Models/*.cpp) \
			$(wildcard Parsers/*.cpp) \
			$(wildcard Writer/*.cpp)
HEADERS = $(wildcard *.h) $(wildcard Models/*.h) $(wildcard Parsers/*.h) $(wildcard Writer/*.h)

parser: $(SOURCES) $(HEADERS)
	g++ $(SOURCES) -o dataparser -std=c++11
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <map>

namespace imagenet
{
	std::map<int, std::string> load_classes(const char* filename)
	{
		std::string line;
		std::ifstream infile(filename);
		std::getline(infile, line);
		std::map<int, std::string> classes;
		while (std::getline(infile, line))
		{
			size_t a = line.find_first_of('\"', 0);
			size_t b = line.find_first_of('\"', a + 1);
			size_t c = line.find_first_of('\"', b + 1);
			size_t d = line.find_first_of('\"', c + 1);
			if (c != d)
			{
				std::string number(line.substr(a + 1, b - a - 1));
				std::string description(line.substr(c + 1, d - c - 1));
				classes[atoi(number.c_str())] = description;
			}
		}
		return classes;
	}
}

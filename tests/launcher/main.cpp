#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.h"

static void
LaunchTests(const char *suiteName, const char *varsPath, const char *testsPath)
{
    TestSuite suite(suiteName);
    ReadConf(suite, varsPath, testsPath);

    suite.launch();
    suite.report();
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        LaunchTests("GMAC", "vars.spec", "tests.spec");
    } else if (argc == 4) {
        LaunchTests(argv[1], argv[2], argv[3]);
    } else {
        std::cerr << "Error: wrong number of parameters" << std::endl;
        std::cerr << " > launcher [ suite_name vars_file tests_file ]" << std::endl;
    }

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

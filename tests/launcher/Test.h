#ifndef TEST_H
#define TEST_H

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <list>
#include <map>
#include <vector>

#include "utils.h"

#include "Variable.h"

class Test
{
private:
    class TestCase {
    public:
        class Stats {
        public:
            typedef std::list<float> ListValues;
            typedef std::list<std::string> ListNames;
        protected:
            typedef std::map<std::string, ListValues> MapStats;
            MapStats stats_;

        public:
            Stats();

            void addPair(std::string name, float val);
            ListNames getNames() const;
            const ListValues &getValues(std::string name) const;
        };

    private:
        typedef std::pair<std::string, std::string> KeyVal;

        struct KeyValFunc : std::unary_function<const KeyVal&, bool>
        {
            const std::string name_;
            explicit KeyValFunc(const std::string& val)
                : name_(val) {}

            bool operator() (const KeyVal &pair)
            {
                return pair.first == name_;
            }
        };


        std::vector<KeyVal> keyvals_;
        std::string name_;
        int status_;
        bool run_;
        double time_;

        void setEnvironment();

    public:
        TestCase();
        TestCase(const TestCase &testCase);
        void addVariableInstance(std::string name, std::string value);

        bool isFailure() const;
        std::string getName() const;
        bool hasVariable(const std::string &name) const;
        bool hasValue(const std::string &name, const std::string &val) const;

        Stats run(const std::string &exec);
        void report(std::ofstream &outfile) const;

        void setElapsedTime(double time);
        double getElapsedTime() const;
    };

    class Stats {
        typedef std::map<const TestCase *, TestCase::Stats> MapStats;
        MapStats testCaseStats_;

    public:
        Stats();
        ~Stats();
        void addTestCaseStats(const TestCase &testCase, const TestCase::Stats &stats);

        Stats filter(Variable &v, std::string value) const;
    };

    Stats stats_;

    std::string name_;
    std::vector<Variable> vars_;

    std::vector<TestCase> testCases_;

    void addTestCase(std::vector<VectorString::iterator> &current);
    bool advance(std::vector<VectorString::iterator> &current,
                 std::vector<VectorString::iterator> &start,
                 std::vector<VectorString::iterator> &end);

    bool equals(std::vector<VectorString::iterator> &current,
                std::vector<VectorString::iterator> &end);

    void generateTestCases();
    
public:
    Test();
    Test(std::string name);

    Test &operator+=(Variable &v);

    std::string getName() const;

    void launch();

    unsigned getNumberOfTestCases() const;
    unsigned getNumberOfFailures() const;

    double getElapsedTime() const;
    
    void report(std::ofstream &outfile) const;
};

class TestSuite {
private:
    std::vector<Test> tests_;
    std::string name_;
    std::ofstream outfile_;

public:
    TestSuite(std::string name);
    ~TestSuite();

    TestSuite &operator+=(Test test);

    void launch();

    unsigned getNumberOfTestCases() const;
    unsigned getNumberOfFailures() const;

    double getElapsedTime() const;

    void report();
};

#endif /* TEST_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

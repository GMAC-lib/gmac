#include "Test.h"

#include <iostream>

Test::TestCase::Stats::Stats()
{}

void
Test::TestCase::Stats::addPair(std::string name, float val)
{
    if (stats_.find(name) == stats_.end()) {
        stats_.insert(MapStats::value_type(name, ListValues()));
    }
    ListValues &vals = stats_[name];
    vals.push_back(val);
    std::cout << name << ":" << val << std::endl;
}

Test::TestCase::Stats::ListNames
Test::TestCase::Stats::getNames() const
{
    ListNames list;
    MapStats::const_iterator it;

    for (it = stats_.begin(); it != stats_.end(); ++it) {
        list.push_back(it->first);
    }

    return list;
}

const Test::TestCase::Stats::ListValues &
Test::TestCase::Stats::getValues(std::string name) const
{
    MapStats::const_iterator it = stats_.find(name);
    assert(it != stats_.end());

    return it->second;
}

Test::TestCase::TestCase() :
    name_(""),
    status_(0),
    run_(false),
    time_(0.0)
{}

Test::TestCase::TestCase(const TestCase &testCase) :
    keyvals_(testCase.keyvals_),
    name_(testCase.name_),
    status_(testCase.status_),
    run_(testCase.run_),
    time_(0.0)
{
}

void
Test::TestCase::addVariableInstance(std::string name, std::string value)
{
    keyvals_.push_back(KeyVal(name, value));
    name_ += name + "=" + value + ";";
}

bool
Test::TestCase::isFailure() const
{
    return status_ != 0;
}

std::string
Test::TestCase::getName() const
{
    return name_;
}

bool
Test::TestCase::hasVariable(const std::string &name) const
{
    std::vector<KeyVal>::const_iterator it = find_if(keyvals_.begin(), keyvals_.end(), KeyValFunc(name));
    return it != keyvals_.end();
}

bool
Test::TestCase::hasValue(const std::string &name, const std::string &value) const
{
    std::vector<KeyVal>::const_iterator it = find_if(keyvals_.begin(), keyvals_.end(), KeyValFunc(name));
    assert(it != keyvals_.end());
    return it->second == value;
}

void
Test::TestCase::report(std::ofstream &outfile) const
{
    outfile << "<testcase ";
    outfile << "name=\"" << name_ << "\" ";
    outfile << "status=\"" << (run_? "run": "notrun") << "\" ";
    outfile << "time=\"" << getElapsedTime() << "\"";
    outfile << ">" << std::endl;
    if (status_ != 0) {
        outfile << "<failure message=\"Exit code: " << status_ << "\"/>" << std::endl;
    }
    outfile << "</testcase>" << std::endl;
}

void
Test::TestCase::setElapsedTime(double time)
{
    time_ = time;
}

double
Test::TestCase::getElapsedTime() const
{
    return time_;
}

Test::Stats::Stats()
{}

Test::Stats::~Stats()
{
    MapStats::iterator it;
    for (it = testCaseStats_.begin(); it != testCaseStats_.end(); ++it) {
        delete it->first;
    }
}

void
Test::Stats::addTestCaseStats(const Test::TestCase &testCase, const Test::TestCase::Stats &stats)
{
    testCaseStats_.insert(MapStats::value_type(new TestCase(testCase), stats));
}


Test::Stats
Test::Stats::filter(Variable &v, std::string value) const
{
    Stats stats;
    assert(v.find(value) != v.end());

    MapStats::const_iterator it;
    for (it = testCaseStats_.begin(); it != testCaseStats_.end(); ++it) {
        const TestCase *testCase = it->first;
        assert(testCase->hasVariable(v.getName()));
        if (testCase->hasValue(v.getName(), value)) {
            stats.addTestCaseStats(*testCase, it->second);
        }
    }

    return stats;
}

void
Test::addTestCase(std::vector<VectorString::iterator> &current)
{
    TestCase conf;

    for (size_t i = 0; i < current.size(); i++) {
        conf.addVariableInstance(vars_[i].getName(), *(current[i]));
    }
    testCases_.push_back(conf);
}

bool
Test::advance(std::vector<VectorString::iterator> &current,
              std::vector<VectorString::iterator> &start,
              std::vector<VectorString::iterator> &end)
{
    if (equals(current, end)) return false;

    for (size_t i = 0; i < current.size(); i++) {
        if ((current[i] + 1) != end[i]) {
            current[i]++;
            break;
        } else {
            if (i != current.size() - 1) {
                current[i] = start[i];
            }
        }
    }

    return true;
}

bool
Test::equals(std::vector<VectorString::iterator> &current,
             std::vector<VectorString::iterator> &end)
{
    for (size_t i = 0; i < current.size(); i++) {
        if ((current[i] + 1) != end[i]) return false;
    }
    return true;
}

void
Test::generateTestCases()
{
    if (vars_.empty() == false) {
        std::vector<VectorString::iterator> start; 
        std::vector<VectorString::iterator> current; 
        std::vector<VectorString::iterator> end; 

        for (size_t i = 0; i < vars_.size(); i++) {
            start.push_back(vars_[i].begin());
            current.push_back(vars_[i].begin());
            end.push_back(vars_[i].end());
        }

        do {
            addTestCase(current);
        } while(advance(current, start, end));
    } else {
        TestCase dummy;
        testCases_.push_back(dummy);
    }
}

Test::Test()
{
}

Test::Test(std::string name) :
    name_(name)
{
}

Test &
Test::operator+=(Variable &v)
{
    vars_.push_back(v);
    return *this;
}

std::string
Test::getName() const
{
    return name_;
}

void
Test::launch()
{
    generateTestCases();

    for (size_t i = 0; i < testCases_.size(); i++) {
        TestCase::Stats stats = testCases_[i].run(name_);
        stats_.addTestCaseStats(testCases_[i], stats);
    }
}

unsigned
Test::getNumberOfTestCases() const
{
    return unsigned(testCases_.size());
}

unsigned
Test::getNumberOfFailures() const
{
    unsigned failures = 0;
    for (size_t i = 0; i < testCases_.size(); i++) {
        if (testCases_[i].isFailure()) failures++; 
    }
    return failures;
}

double
Test::getElapsedTime() const
{
    double elapsedTime = 0;
    for (size_t i = 0; i < testCases_.size(); i++) {
        elapsedTime += testCases_[i].getElapsedTime();
    }
    return elapsedTime;
}

void
Test::report(std::ofstream &outfile) const
{
    outfile << "<testsuite ";
    outfile << "name=\"" << name_ << "\" ";
    outfile << "tests=\"" << getNumberOfTestCases() << "\" ";
    outfile << "failures=\"" << getNumberOfFailures() << "\" ";
    outfile << "time=\"" << getElapsedTime() << "\" ";
    outfile << "errors=\"0\" ";
    outfile << ">" << std::endl;

    for (size_t i = 0; i < testCases_.size(); i++) {
        testCases_[i].report(outfile);
    }

    outfile << "</testsuite>" << std::endl;
}

TestSuite::TestSuite(std::string name) :
    name_(name),
    outfile_((name + ".report.xml").c_str(), std::ofstream::out | std::ofstream::trunc)
{
}

TestSuite::~TestSuite()
{
    outfile_.close();
}

TestSuite &
TestSuite::operator+=(Test test)
{
    tests_.push_back(test);
    return *this;
}

void
TestSuite::launch()
{
    for (size_t i = 0; i < tests_.size(); i++) {
        tests_[i].launch();
    }
}

unsigned
TestSuite::getNumberOfTestCases() const
{
    unsigned testCases = 0;
    for (size_t i = 0; i < tests_.size(); i++) {
        testCases += tests_[i].getNumberOfTestCases();
    }
    return testCases;
}

unsigned
TestSuite::getNumberOfFailures() const
{
    unsigned failures = 0;
    for (size_t i = 0; i < tests_.size(); i++) {
        failures += tests_[i].getNumberOfFailures();
    }
    return failures;
}

double
TestSuite::getElapsedTime() const
{
    double elapsedTime = 0;
    for (size_t i = 0; i < tests_.size(); i++) {
        elapsedTime += tests_[i].getElapsedTime();
    }
    return elapsedTime;
}

void
TestSuite::report()
{
    outfile_ << std::boolalpha;
    outfile_ << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;

    outfile_ << "<testsuites ";
    outfile_ << "name=\"" << name_ << "\" ";
    outfile_ << "tests=\"" << getNumberOfTestCases() << "\" ";
    outfile_ << "failures=\"" << getNumberOfFailures() << "\" ";
    outfile_ << "time=\"" << getElapsedTime() << "\" ";
    outfile_ << "errors=\"0\"";
    outfile_ << ">" << std::endl;

    for (size_t i = 0; i < tests_.size(); i++) {
        tests_[i].report(outfile_);
    }

    outfile_ << "</testsuites>" << std::endl;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */

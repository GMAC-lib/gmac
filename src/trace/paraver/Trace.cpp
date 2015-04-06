#include "Trace.h"

#include "util/Logger.h"

#include <time.h>
#include <iomanip>
#include <sstream>

namespace __impl { namespace trace { namespace paraver {

#if defined(_MSC_VER)
static struct tm *gmac_localtime(time_t *a)
{
    static struct tm time_struct;
    localtime_s(&time_struct, a);
    return &time_struct;
}
#else
#   define gmac_localtime(a) localtime(a)
#endif

void TraceWriter::processTrace(Thread *thread, uint64_t t, StateName *state)
{
	static bool idle = false;
	if(state->getValue() == 0) {
		thread->start(of_, state->getValue(), t);
		idle = true;
	}
	else if(idle == true) {
		thread->end(of_, t);
		idle = false;
	}
}

TraceWriter::TraceWriter(const char *fileName, uint32_t pid, uint32_t tid) :
    of_(fileName)
{
	// Create the root application and add the current task
	apps_.push_back(new Application(1, "app"));
	Task *task = apps_.back()->addTask(pid);
	task->addThread(tid);
}

TraceWriter::~TraceWriter()
{
    std::list<Application *>::iterator i;
    for(i = apps_.begin(); i != apps_.end(); ++i) delete (*i);
    apps_.clear();
}

void TraceWriter::addThread(uint32_t pid, uint32_t tid)
{
    ASSERTION(apps_.empty() == false);
    mutex_.lock();
    Task *task = apps_.back()->getTask(pid);
    task->addThread(tid);
    mutex_.unlock();
}

void TraceWriter::addTask(uint32_t pid)
{
    mutex_.lock();
    apps_.back()->addTask(pid);
    mutex_.unlock();
}

void TraceWriter::pushState(uint64_t t, uint32_t pid, uint32_t tid,
		const StateName &state)
{
    mutex_.lock();
	Task *task = apps_.back()->getTask(pid);
    if(task != NULL) {
    	Thread *thread = task->getThread(tid);
        if(thread != NULL) {
            thread->end(of_, t);
        	thread->start(of_, state.getValue(), t);
        }
    }
    mutex_.unlock();
}


void TraceWriter::pushEvent(uint64_t t, uint32_t pid, uint32_t tid,
		uint64_t ev, int64_t value)
{
    mutex_.lock();
	Task *task = apps_.back()->getTask(pid);
    if(task != NULL) {
        Thread *thread = task->getThread(tid);
        if(thread != NULL) {
    	    Event event(thread, t, ev, value);
        	event.write(of_);
        }
    }
    mutex_.unlock();
}


void TraceWriter::pushEvent(uint64_t t, uint32_t pid, uint32_t tid,
        const EventName &event, int64_t value)
{
    pushEvent(t, pid, tid, event.getValue(), value);
}

void TraceWriter::pushCommunication(uint64_t start, uint32_t srcPid, uint32_t srcTid,
        uint64_t end, uint32_t dstPid, uint32_t dstTid, uint64_t size)
{
    mutex_.lock();
    Task *srcTask = apps_.back()->getTask(srcPid);
    if(srcTask == NULL) { mutex_.unlock(); return; }
    Task *dstTask = apps_.back()->getTask(dstPid);
    if(dstTask == NULL) { mutex_.unlock(); return; }
    Thread *srcThread = srcTask->getThread(srcTid);
    if(srcThread == NULL) { mutex_.unlock(); return; }
    Thread *dstThread = dstTask->getThread(dstTid);
    if(dstThread == NULL) { mutex_.unlock(); return; }
    Communication comm(srcThread, dstThread, start, end, size);
    comm.write(of_);
    mutex_.unlock();
}

void TraceWriter::write(uint64_t t)
{
    mutex_.lock();
	std::list<Application *>::iterator app;
	for(app = apps_.begin(); app != apps_.end(); ++app) {
		(*app)->end(of_, t);
	}

	Record::end(of_);
	uint32_t size = uint32_t(apps_.size());
	of_.write((char *)&size, sizeof(size));
	for(app = apps_.begin(); app != apps_.end(); ++app) {
		(*app)->write(of_);
	}

	of_.close();
    mutex_.unlock();
}


void TraceReader::buildApp(std::ifstream &in)
{
	uint32_t id, nTasks;
	in.read((char *)&id, sizeof(id));
	in.read((char *)&nTasks, sizeof(nTasks));

	std::ostringstream name;
	name << "App" << id;
	apps_.push_back(new Application(id, name.str()));

	for(uint32_t i = 0; i < nTasks; i++) {
		uint32_t nThreads;
		in.read((char *)&id, sizeof(id));
		in.read((char *)&nThreads, sizeof(nThreads));
		Task *task = apps_.back()->addTask(id);
		for(uint32_t j = 0; j < nThreads; j++) {
			uint32_t dummy;
			in.read((char *)&id, sizeof(id));
			in.read((char *)&dummy, sizeof(dummy));
			task->addThread(id);
		}
	}
}

TraceReader::TraceReader(const char *fileName) :
    endTime_(0)
{
	std::ifstream in;
	in.open(fileName, std::ios::binary);
    ASSERTION(in.is_open() != 0);
	// Read records from file
	Record *record = NULL;
	while((record = Record::read(in)) != NULL) {
		records_.push_back(record);
		endTime_ = (endTime_ > record->getEndTime()) ? endTime_ : record->getEndTime();
	} 

	// Sort the records
	records_.sort(RecordPredicate());

	// Read header
	uint32_t nApps;
	in.read((char *)&nApps, sizeof(nApps));
	for(uint32_t i = 0; i < nApps; i++) buildApp(in);
}

StreamOut &operator<<(StreamOut &of, const TraceReader &trace)
{
	time_t timep = time(NULL);
	struct tm *t = gmac_localtime(&timep);
	int32_t year=(t->tm_year<100)?t->tm_year:t->tm_year-100;
	// Print the file header: date/time
	of << "#Paraver(";
	of << std::setw(2) << std::setfill('0') << t->tm_mday << "/";
	of << std::setw(2) << std::setfill('0') << t->tm_mon << "/";
	of << std::setw(2) << std::setfill('0') << year;
	of << " at " << std::setw(2) << std::setfill('0') << t->tm_hour << ":";
	of << std::setw(2) << std::setfill('0') << t->tm_min << ")";
	of << ":" << trace.endTime_; 

	// Without resource mode
	of << ":" << 0;

	// Print # of applications and tasks and threads per task
	of << ":" << trace.apps_.size();
	std::list<Application *>::const_iterator app;
	for(app = trace.apps_.begin(); app != trace.apps_.end(); ++app)
		of << ":" << *(*app);
	of << std::endl;

	std::list<Record *>::const_iterator i;
	for(i = trace.records_.begin(); i != trace.records_.end(); ++i)
		of << *(*i);
    return of;
}



} } }

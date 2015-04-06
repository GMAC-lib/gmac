#include "Handler.h"

namespace __impl { namespace memory {
Handler *Handler::Handler_ = NULL;

Handler::CallBack Handler::Entry_ = NULL;
Handler::CallBack Handler::Exit_ = NULL;

} };

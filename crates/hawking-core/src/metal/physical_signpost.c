#include <os/log.h>
#include <os/signpost.h>
#include <pthread.h>
#include <stdint.h>

static os_log_t hawking_log;
static pthread_once_t hawking_log_once = PTHREAD_ONCE_INIT;

static void hawking_physical_log_init(void) {
    hawking_log = os_log_create("org.hawking.physical-evidence", "metal-phase");
}

static os_log_t hawking_physical_log(void) {
    (void)pthread_once(&hawking_log_once, hawking_physical_log_init);
    return hawking_log;
}

void hawking_physical_signpost_begin(uint64_t interval_id, const char *identity) {
    os_signpost_interval_begin(
        hawking_physical_log(), interval_id, "HawkingPhysicalPhase", "%{public}s", identity
    );
}

void hawking_physical_signpost_end(uint64_t interval_id, const char *identity) {
    os_signpost_interval_end(
        hawking_physical_log(), interval_id, "HawkingPhysicalPhase", "%{public}s", identity
    );
}

// Temporary rollback switch; remove after the meeting pipeline changes are validated in production.
export const meetingPerformanceEnabled =
  import.meta.env.VITE_DISABLE_MEETING_PERF_OPTIMIZATIONS !== "true";

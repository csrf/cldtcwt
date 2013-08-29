#
# Regular cron jobs for the cldtcwt package
#
0 4	* * *	root	[ -x /usr/bin/cldtcwt_maintenance ] && /usr/bin/cldtcwt_maintenance

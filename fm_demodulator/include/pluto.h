#ifndef PLUTO_H
#define PLUTO_H

#include <iio.h>

#include "common.h"

bool pluto_init(const char *uri);
void pluto_cleanup(void);
bool pluto_configure_stream(struct stream_cfg *cfg);
bool pluto_start_streaming(void);
ssize_t pluto_read_samples(int16_t **i_samples, int16_t **q_samples);

void errchk(int v, const char *what);
void wr_ch_lli(struct iio_channel *chn, const char *what, long long val);
void wr_ch_str(struct iio_channel *chn, const char *what, const char *str);
void wr_ch_double(struct iio_channel *chn, const char *what, double val);

char *get_ch_name(const char *type, int id);
struct iio_device *get_ad9361_phy(void);
bool get_ad9361_stream_dev(enum iodev d, struct iio_device **dev);
bool get_ad9361_stream_ch(enum iodev d, struct iio_device *dev, int chid,
                          struct iio_channel **chn);
bool get_phy_chan(enum iodev d, int chid, struct iio_channel **chn);
bool get_lo_chan(enum iodev d, struct iio_channel **chn);
bool cfg_ad9361_gain_control(enum iodev type, int chid);
bool cfg_ad9361_streaming_ch(struct stream_cfg *cfg, enum iodev type, int chid);

#endif
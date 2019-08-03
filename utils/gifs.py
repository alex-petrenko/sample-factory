from subprocess import Popen, PIPE


def encode_gif(images, fps):
    if len(images[0].shape) == 3:
        h, w, c = images[0].shape
    else:
        h, w = images[0].shape
        c = 1
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (w, h),
           '-pix_fmt', {1: 'gray', 3: 'rgb24', 4: 'rgba'}[c],
           '-i', '-',
           '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '-f', 'gif',
           '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out

#ifndef VBOBUFFER_H
#define VBOBUFFER_H


#include <GL/gl.h>
#include <vector>

// Vertex buffer object holder, that will automatically create or remove them
class VBOBuffers {
public:
	
    VBOBuffers(const VBOBuffers&) = default;
	VBOBuffers(int num = 0);
	~VBOBuffers();

	GLuint getBuffer(int n);

private:

	std::vector<GLuint> buffers_;

};



#endif


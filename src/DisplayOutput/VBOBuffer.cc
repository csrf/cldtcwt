#include "VBOBuffer.h"
    
VBOBuffers::VBOBuffers(int num)
 : buffers_(num)
{
	glGenBuffers(num, &buffers_[0]);
}


GLuint VBOBuffers::getBuffer(int n)
{
	return buffers_[n];
}


VBOBuffers::~VBOBuffers()
{
	glDeleteBuffers(buffers_.size(), &buffers_[0]);
}



#version 420

layout(location = 0) out vec4 fragmentColor;
in vec2 texCoordOut;

uniform sampler2D texs0;
uniform sampler2D texs1;



void main()
{
	vec4 color0 = texture(texs0, texCoordOut);
	vec4 color1 = texture(texs1, texCoordOut);
	if(color0.w == 0)
		fragmentColor = color1;
	else if(color1.w == 0)
		fragmentColor = color0;
	else
		fragmentColor = (texture(texs0, texCoordOut) * 0.5 + texture(texs1, texCoordOut) * 0.5);

}
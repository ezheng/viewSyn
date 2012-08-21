#version 420

layout(location = 0) out vec4 fragmentColor;
in vec2 texCoordOut;

uniform sampler2D texs0;
uniform sampler2D texs1;
uniform float weight;
uniform float x_texSize;
uniform float y_texSize;

bool getColor( in vec2 texCoord, out vec4 color)
{
	vec4 color0 = texture(texs0, texCoord);
	vec4 color1 = texture(texs1, texCoord);
	if(color0.w !=0 && color1.w!=0)
	{
		color = mix(color0, color1, 1 - weight);
		return true;
	}
	else if(color0.w != 0)
	{
		color = color0;
		return true;
	}
	else if(color1.w != 0)
	{
		color = color1;
		return true;	
	}	
	else
	{
		//color = vec4(0,0,0,0);
		return false;
	}
}


void main()
{
	if(getColor( texCoordOut, fragmentColor))
		return;
		
	for(int layer = 1; layer < 50; layer++)
	{
		float cornerOffsetX = layer * x_texSize;
		float cornerOffsetY = layer * y_texSize;
		for(int i = 0; i < layer; i++)
		{
			float offsetX = i * x_texSize;
			float offsetY = i * y_texSize;

			if(getColor( vec2( texCoordOut.x - cornerOffsetX , texCoordOut.y + offsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x - cornerOffsetX, texCoordOut.y - offsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x + cornerOffsetX, texCoordOut.y + offsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x + cornerOffsetX, texCoordOut.y - offsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x + offsetX, texCoordOut.y + cornerOffsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x + offsetX, texCoordOut.y - cornerOffsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x - offsetX, texCoordOut.y + cornerOffsetY), fragmentColor))
				return;
			if(getColor( vec2( texCoordOut.x - offsetX, texCoordOut.y - cornerOffsetY), fragmentColor))
				return;
		}
		// four courners
		if(getColor( vec2( texCoordOut.x - cornerOffsetX , texCoordOut.y + cornerOffsetY), fragmentColor))
			return;
		if(getColor( vec2( texCoordOut.x - cornerOffsetX, texCoordOut.y - cornerOffsetY), fragmentColor))
			return;
		if(getColor( vec2( texCoordOut.x + cornerOffsetX, texCoordOut.y + cornerOffsetY), fragmentColor))
			return;
		if(getColor( vec2( texCoordOut.x + cornerOffsetX, texCoordOut.y - cornerOffsetY), fragmentColor))
			return;
	}
}
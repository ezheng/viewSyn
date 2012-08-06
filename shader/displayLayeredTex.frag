#version 420

layout(location = 0) out vec4 fragmentColor;
in vec2 texCoordOut;

uniform sampler2D texs0;
uniform sampler2D texs1;
uniform float weight;
uniform float texSize;


void main()
{
	vec4 color0 = texture(texs0, texCoordOut);
	vec4 color1 = texture(texs1, texCoordOut);
	if(color0.w == 0 && color1.w != 0)
		fragmentColor = color1;
	else if(color0.w != 0 && color1.w == 0)
		fragmentColor = color0;
	else if(color0.w == 0 && color1.w == 0)
	{
		for(int i = 1; i < 50; i++)
		{
			vec4 color2 = texture(texs0, vec2(texCoordOut.x + texSize * float(i), texCoordOut.y));
			vec4 color3 = texture(texs1, vec2(texCoordOut.x + texSize * float(i), texCoordOut.y));
			if(color2.w !=0 && color3.w!=0)
			{
				fragmentColor = mix(color2, color3, 1 - weight);
				break;
			}
			else if(color2.w != 0)
			{
				fragmentColor = color2;
				break;
			}
			else if(color3.w != 0)
			{
				fragmentColor = color3;
				break;	
			}

			color2 = texture(texs0, vec2(texCoordOut.x - texSize * float(i), texCoordOut.y));
			color3 = texture(texs1, vec2(texCoordOut.x - texSize * float(i), texCoordOut.y));
			if(color2.w !=0 && color3.w!=0)
			{
				fragmentColor = mix(color2, color3, 1 - weight);
				break;
			}
			else if(color2.w != 0)
			{
				fragmentColor = color2;
				break;
			}
			else if(color3.w != 0)
			{
				fragmentColor = color3;
				break;	
			}
		}
	}

	else 
		//fragmentColor = (texture(texs0, texCoordOut) * 0.5 + texture(texs1, texCoordOut) * 0.5);
		fragmentColor = mix(color0, color1, 1 - weight);
}
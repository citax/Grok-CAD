#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;

uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uEye;
uniform float uSelected;
uniform float uAlpha;

out vec4 FragColor;

void main() {
  vec3 n = normalize(vNormal);
  vec3 l = normalize(-uLightDir);
  float ndotl = max(dot(n, l), 0.0);
  float ndotl2 = max(dot(-n, l), 0.0);
  float lit = max(ndotl, ndotl2 * 0.35);
  vec3 view = normalize(uEye - vWorldPos);
  vec3 h = normalize(l + view);
  float spec = pow(max(dot(n, h), 0.0), 32.0);
  vec3 base = uColor;
  if (uSelected > 0.5) {
    base = mix(base, vec3(1.0, 0.9, 0.25), 0.55);
  }
  vec3 ambient = 0.28 * base;
  vec3 diffuse = 0.62 * lit * base;
  vec3 specular = 0.12 * spec * vec3(1.0);
  FragColor = vec4(ambient + diffuse + specular, uAlpha);
}

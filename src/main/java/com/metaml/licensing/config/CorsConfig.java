package com.metaml.licensing.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {
  @Override
  public void addCorsMappings(CorsRegistry registry) {
    registry.addMapping("/**")
      .allowedOrigins(
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://208.109.36.23:5173"
      )
      .allowedMethods("GET","POST","PUT","DELETE","OPTIONS")
      .allowedHeaders("Content-Type","Authorization","X-User","X-Password")
      .allowCredentials(false)
      .maxAge(3600);
  }
}

import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/dist/ScrollTrigger'

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger)
}

export function useScrollReveal(containerRef) {
  useEffect(() => {
    if (!containerRef.current) return

    const cards = containerRef.current.querySelectorAll('.chart-card')
    const findings = containerRef.current.querySelectorAll('.finding-block')

    const elements = [...cards, ...findings]

    elements.forEach((el, i) => {
      gsap.fromTo(
        el,
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 85%',
            end: 'top 50%',
            toggleActions: 'play none none reverse',
          },
          delay: i * 0.1,
        }
      )
    })

    return () => {
      ScrollTrigger.getAll().forEach(t => t.kill())
    }
  }, [containerRef])
}
